"""lerobot-rollout with the mixed-camera-size Diffusion patch.

Same arguments as lerobot-rollout. Diffusion stacks the camera images
before the resize, so the 848x480 top and 640x480 wrist/side views need
patch_diffusion() (see train_mixed.py). Other policies are unchanged.

fast_prepare() replaces prepare_observation_for_inference: the stock
one converts the 3 camera images to float on the CPU every tick, which
competes with the video encoder threads (loop 10-15 Hz, not 30 Hz).
Here the uint8 images go to the GPU first and are converted there.

PREVIEW=1 (default) shows the cameras in one mpv window, as in
record_preview.py, with the control loop rate. Keys go to LeRobot's
global listener (Right, Left, Esc), with any window in focus.
Env: PREVIEW, PREVIEW_W (default 1600), PREVIEW_FPS (default 15),
PREVIEW_LABEL (text on the window), SMOOTH (SmolVLA chunk moving
average over this many actions; 0 = off), FREEZE=1 (no autograd in
RTC guidance), COMPILE=default (compiled denoise step),
RTC_EXACT_DELAY=1 (RTC skips the actions really used),
ROLLOUT_DEBUG (.npz log path, read with rollout_debug.py),
LIVE_SKILL=1 (keys 1/2/3 set the task text of the next plans to the
box/bag/top-box skill text, 4/5 far edge/on table if the split file
has them, 0 = back to the --task text; the preview
shows the skill), SKILL_SPLIT (split file of make_skill3.py with the
texts; default: the base texts), SKILL_LOG (CSV of the key times and
skills), PLANNER=log|drive (box/bag planner thread of
planner/live_planner.py; log: calls only, drive: it sets the text;
needs LIVE_SKILL=1), PLANNER_PROBES (default planner/planner_500m.npz),
PLANNER_LOG (CSV of the planner calls).
"""

import collections
import os
import time

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from parcel_vla.training.train_mixed import patch_diffusion

# key: (skill, task text); the texts of make_skill3.py
SKILLS = {"1": ("box", "pick the box and place it in the flip zone"),
          "2": ("bag", "pick the bag and place it in the flip zone"),
          "3": ("top", "pick the top box off the stack")}
LIVE = {"skill": "", "task": None}


def fast_prepare(observation, device, task=None, robot_type=None):
    for name in observation:
        x = torch.from_numpy(observation[name]).to(device, non_blocking=True)
        if "image" in name:
            if x.dtype == torch.uint8:
                x = x.float().div_(255)
            x = x.permute(2, 0, 1).contiguous()
        observation[name] = x.unsqueeze(0)
    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return observation


def patch_prepare():
    from lerobot.rollout.inference import rtc, sync
    for mod in (sync, rtc):
        if hasattr(mod, "prepare_observation_for_inference"):
            mod.prepare_observation_for_inference = fast_prepare


def patch_smooth(k):
    """Moving average over k actions of each SmolVLA chunk (k odd).

    SmolVLA 20k chunks are ~20x rougher than the demos (2nd difference);
    k=7 cuts that ~8x with no loss of offline accuracy.
    """
    import torch.nn.functional as F
    from lerobot.policies.smolvla import modeling_smolvla as ms
    orig = ms.SmolVLAPolicy._get_action_chunk

    def smoothed(self, *a, **kw):
        x = orig(self, *a, **kw)  # (B, T, D)
        y = F.pad(x.transpose(1, 2).float(), (k // 2, k // 2),
                  mode="replicate")
        y = F.avg_pool1d(y, k, stride=1).transpose(1, 2)
        return y.to(x.dtype)

    ms.SmolVLAPolicy._get_action_chunk = smoothed


def patch_first_plan(freeze, compile_mode):
    """Speed-ups on the first SmolVLA plan (the policy exists then).

    freeze: requires_grad off for all parameters. RTC guidance runs the
    denoising under enable_grad and records an unused autograd graph:
    plan 273 -> 204 ms offline, same output.
    compile_mode: inference_improvement/fast_denoise.py on the denoise
    step ("default").
    The first plan then compiles both paths (no guidance, RTC guidance;
    4-20 s, the robot waits) and the RTC latency tracker drops it: its
    max would set the delay for the run.
    """
    from parcel_vla.inference_improvement import fast_denoise
    from lerobot.policies.rtc.latency_tracker import LatencyTracker
    from lerobot.policies.smolvla import modeling_smolvla as ms
    orig = ms.SmolVLAPolicy._get_action_chunk
    skip = [0]

    def chunk(self, *a, **kw):
        if not getattr(self, "_first_plan_done", False):
            self._first_plan_done = True
            if freeze:
                self.requires_grad_(False)
            if compile_mode:
                fast_denoise.enable(self.model, compile_mode)
                t = time.perf_counter()
                out = orig(self, *a, **kw)
                if self._rtc_enabled():  # guidance path: own graph
                    orig(self, a[0], None, inference_delay=5,
                         prev_chunk_left_over=out[0, 20:])
                skip[0] = 1
                print(f"[compile] first plan {time.perf_counter() - t:.1f}"
                      " s", flush=True)
        return orig(self, *a, **kw)

    add = LatencyTracker.add

    def add_skip_compile(self, latency):
        if skip[0]:
            skip[0] = 0
            return
        add(self, latency)

    ms.SmolVLAPolicy._get_action_chunk = chunk
    LatencyTracker.add = add_skip_compile


def patch_rtc_exact_delay():
    """RTC queue: skip the actions really used during the plan.

    LeRobot skips real_delay = ceil(latency x fps) actions of the new
    chunk; this is often 1 more than the arm used, so the arm jumps one
    action ahead at each switch.
    """
    from lerobot.policies.rtc.action_queue import ActionQueue
    check = ActionQueue._check_and_resolve_delays

    def resolve(self, real_delay, action_index_before_inference=None):
        if action_index_before_inference is None:
            return check(self, real_delay)
        return max(0, self.last_index - action_index_before_inference)

    ActionQueue._check_and_resolve_delays = resolve


def patch_rtc_fresh_obs():
    """RTC reset: also drop the last observation.

    The stock reset clears the action queue but keeps the observation
    of the last autonomous tick. After a DAgger pause and correction,
    the first plan then uses the old scene and the arm goes back to the
    pose of the pause. With the observation cleared, the RTC thread
    waits for the first new tick.
    """
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    reset = RTCInferenceEngine.reset

    def fresh_reset(self):
        reset(self)
        with self._obs_lock:
            self._obs_holder["obs"] = None

    RTCInferenceEngine.reset = fresh_reset


def patch_leader_torque(retries=5):
    """Leader arm torque on/off: retry, then warn instead of a crash.

    The stock calls try once. One bad status packet from a leader motor
    at a DAgger handover stopped the whole session. Leader torque only
    holds the arm between corrections, so a failed switch is not unsafe.
    """
    import logging
    from lerobot.teleoperators.so_leader.so_leader import SOLeader
    log = logging.getLogger("leader_torque")

    def make(name):
        def call(self):
            try:
                getattr(self.bus, name)(num_retry=retries)
            except ConnectionError as e:
                log.warning("leader %s failed after %d retries: %s",
                            name, retries, e)
        return call

    SOLeader.enable_torque = make("enable_torque")
    SOLeader.disable_torque = make("disable_torque")


def patch_debug(path, label=""):
    """Log the rollout to path (.npz) and print one line per plan.

    plan: start / end time of each SmolVLA chunk (sync and RTC).
    send: time and joint command of each tick. obs: time and measured
    joint position. merge (RTC only): real_delay, actions used during
    the plan, rest of the old chunk, full new chunk. Saved after each
    plan. Read with rollout_debug.py.
    """
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.smolvla import modeling_smolvla as ms
    from lerobot.rollout import context
    log = {"plan": [], "send": [], "obs": [], "merge": []}
    keys = []

    def save():
        arr = {}
        for name in ("send", "obs"):
            rows = list(log[name])
            arr[f"{name}_t"] = np.array([r[0] for r in rows])
            arr[f"{name}_q"] = np.array([r[1] for r in rows])
        arr["plan"] = np.array(list(log["plan"]))
        arr["merge"] = np.array(list(log["merge"]), dtype=object)
        np.savez(path, keys=np.array(keys), label=label, **arr)

    get_chunk = ms.SmolVLAPolicy._get_action_chunk

    def chunk(self, *a, **kw):
        t0 = time.perf_counter()
        out = get_chunk(self, *a, **kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        log["plan"].append((t0, t1))
        sends = log["send"]
        gap = (t1 - sends[-1][0]) * 1000 if sends else 0.0
        print(f"[debug] plan {len(log['plan']):4d}  "
              f"{(t1 - t0) * 1000:5.0f} ms  last send {gap:5.0f} ms ago",
              flush=True)
        save()
        return out

    merge = ActionQueue.merge

    def logged_merge(self, original, processed, real_delay,
                     action_index_before_inference=None):
        old = None if self.queue is None else \
            self.queue[self.last_index:].float().cpu().numpy()
        used = -1 if action_index_before_inference is None else \
            self.last_index - action_index_before_inference
        merge(self, original, processed, real_delay,
              action_index_before_inference)
        log["merge"].append((time.perf_counter(), real_delay, used, old,
                             processed.float().cpu().numpy()))
        print(f"[debug]   rtc merge: real_delay {real_delay}  used {used}"
              f"  old chunk left {0 if old is None else len(old)}",
              flush=True)
        save()

    make = context.make_robot_from_config

    def make_logged(cfg):
        robot = make(cfg)
        send, observe = robot.send_action, robot.get_observation

        def send_action(action, *a, **kw):
            if not keys:
                keys.extend(k for k in action if k.endswith(".pos"))
            log["send"].append((time.perf_counter(),
                                [float(action[k]) for k in keys]))
            return send(action, *a, **kw)

        def get_observation(*a, **kw):
            obs = observe(*a, **kw)
            if keys:
                log["obs"].append((time.perf_counter(),
                                   [float(obs[k]) for k in keys]))
            return obs

        robot.send_action, robot.get_observation = send_action, \
            get_observation
        return robot

    ms.SmolVLAPolicy._get_action_chunk = chunk
    ActionQueue.merge = logged_merge
    context.make_robot_from_config = make_logged
    import atexit
    atexit.register(save)
    print(f"[debug] log -> {path}", flush=True)


def patch_live_skill(own_listener, log_path=None):
    """Keys 1/2/3 select the task text of the next plans; 0 = --task.

    The sync and RTC engines read self._task at each plan, so _task
    becomes a property that returns the live text. own_listener: start
    a key listener (base strategy has none); else the digit keys are
    added to the strategy's listener (one listener on the terminal).
    """
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    from lerobot.rollout.inference.sync import SyncInferenceEngine
    from lerobot.utils import keyboard_input as ki

    def get(self):
        return LIVE["task"] or self.__dict__["_task_cfg"]

    def put(self, value):
        self.__dict__["_task_cfg"] = value

    for cls in (RTCInferenceEngine, SyncInferenceEngine):
        cls._task = property(get, put)

    def on_key(name):
        if name == "0":
            LIVE.update(skill="", task=None)
        elif name in SKILLS:
            LIVE["skill"], LIVE["task"] = SKILLS[name]
        else:
            return False
        print(f"[skill] {LIVE['skill'] or 'start text'}", flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"{time.time():.3f},{LIVE['skill'] or 'start'}\n")
        return True

    help_text = ", ".join(f"{k} {s}" for k, (s, _) in SKILLS.items()) \
        + ", 0 start text"
    if own_listener:
        ki.create_key_listener(on_key, controls_help=help_text)
    else:
        make = ki.create_key_listener

        def with_skills(dispatch, controls_help=""):
            return make(lambda n: on_key(n) or dispatch(n),
                        controls_help=f"{controls_help}; {help_text}")

        ki.create_key_listener = with_skills
    print(f"[skill] keys: {help_text}", flush=True)


def patch_planner(drive, probes, log_path):
    """Box/bag planner thread on the robot observations (see
    planner/live_planner.py). drive: it also sets the skill text."""
    from parcel_vla.planner.live_planner import LivePlanner
    from lerobot.rollout import context
    start = os.environ.get("SKILL", "start")

    def set_skill(name):
        key = next(k for k, (s, _) in SKILLS.items() if s == name)
        LIVE["skill"], LIVE["task"] = SKILLS[key]

    planner = LivePlanner(probes, log_path, drive, set_skill,
                          lambda: LIVE["skill"] or start)
    make = context.make_robot_from_config

    def make_with_planner(cfg):
        robot = make(cfg)
        observe = robot.get_observation

        def get_observation(*a, **kw):
            obs = observe(*a, **kw)
            planner.update(obs)
            return obs

        robot.get_observation = get_observation
        planner.start()
        return robot

    context.make_robot_from_config = make_with_planner


def start_preview(robot, label=None):
    """Camera preview of a connected robot, with its send_action rate.

    Does not take the keyboard focus (the terminal keeps it).
    """
    # record_preview patches LeRobotDataset for phase marks on import;
    # undo that, only its Preview window is used here
    originals = (LeRobotDataset.add_frame, LeRobotDataset.save_episode,
                 LeRobotDataset.clear_episode_buffer)
    from parcel_vla.data import record_preview as rp
    (LeRobotDataset.add_frame, LeRobotDataset.save_episode,
     LeRobotDataset.clear_episode_buffer) = originals

    ticks = collections.deque(maxlen=60)
    label = label or os.environ.get("PREVIEW_LABEL", "rollout")

    class RolloutPreview(rp.Preview):
        def _compose(self):
            canvas = np.zeros((self.h, self.w, 3), np.uint8)
            names = list(self.cams)
            top = rp.even((self.h - self.big_h) / 2)
            tiles = [(names[0], 0, top, self.big_w, self.big_h)]
            tiles += [(n, self.big_w, i * self.small_h, self.small_w,
                       self.small_h) for i, n in enumerate(names[1:])]
            for name, x, y, w, h in tiles:
                canvas[y:y + h, x:x + w] = self._frame(self.cams[name], w, h)
                cv2.putText(canvas, name, (x + 10, y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            hz = 0.0
            if len(ticks) > 1 and time.time() - ticks[-1] < 0.5:
                hz = (len(ticks) - 1) / (ticks[-1] - ticks[0])
            colour = (0, 255, 0) if hz >= 27 else (255, 60, 60)
            text = f"{label}   loop {hz:4.1f} Hz" if hz else \
                f"{label}   robot idle"
            cv2.putText(canvas, text, (10, self.h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
            if os.environ.get("LIVE_SKILL", "0") != "0":
                cv2.putText(canvas, f"skill: {LIVE['skill'] or 'start'}",
                            (10, self.h - 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, (0, 200, 255), 4)
            return canvas

        def run(self):
            self._take_focus = lambda: None
            super().run()

    send = robot.send_action

    def send_action(action, *a, **kw):
        ticks.append(time.time())
        return send(action, *a, **kw)

    robot.send_action = send_action
    if robot.cameras:
        RolloutPreview(robot.cameras,
                       int(os.environ.get("PREVIEW_W", 1600)),
                       int(os.environ.get("PREVIEW_FPS", 15))).start()


def patch_preview():
    from lerobot.rollout import context
    make = context.make_robot_from_config

    def make_robot_with_preview(cfg):
        robot = make(cfg)
        start_preview(robot)
        return robot

    context.make_robot_from_config = make_robot_with_preview


if __name__ == "__main__":
    patch_diffusion()
    patch_prepare()
    patch_rtc_fresh_obs()
    patch_leader_torque()
    if int(os.environ.get("SMOOTH", "0")) > 1:
        patch_smooth(int(os.environ["SMOOTH"]) // 2 * 2 + 1)
    if os.environ.get("PREVIEW", "1") != "0":
        patch_preview()
    if os.environ.get("FREEZE", "0") != "0" or os.environ.get("COMPILE"):
        patch_first_plan(os.environ.get("FREEZE", "0") != "0",
                         os.environ.get("COMPILE"))
    if os.environ.get("RTC_EXACT_DELAY", "0") != "0":
        patch_rtc_exact_delay()
    if os.environ.get("ROLLOUT_DEBUG"):
        patch_debug(os.environ["ROLLOUT_DEBUG"],
                    os.environ.get("ROLLOUT_DEBUG_LABEL", ""))
    if os.environ.get("LIVE_SKILL", "0") != "0":
        import json
        import sys
        if os.environ.get("SKILL_SPLIT"):
            with open(os.environ["SKILL_SPLIT"]) as f:
                texts = json.load(f)["texts"]
            SKILLS.update({k: (s, texts[s]) for k, (s, _) in SKILLS.items()})
            for key, name in (("4", "far"), ("5", "push")):
                if name in texts:
                    SKILLS[key] = (name, texts[name])
        patch_live_skill("--strategy.type=base" in sys.argv,
                         os.environ.get("SKILL_LOG"))
    if os.environ.get("PLANNER"):
        from pathlib import Path
        default = Path(__file__).parents[1] / "planner/planner_500m.npz"
        patch_planner(os.environ["PLANNER"] == "drive",
                      os.environ.get("PLANNER_PROBES", str(default)),
                      os.environ["PLANNER_LOG"])
    from lerobot.scripts.lerobot_rollout import main
    main()
