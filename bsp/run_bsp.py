"""Run an ACT-BSP or Diffusion-BSP policy on the SO-101 (BSP executor).

Two threads share the robot (one lock for the motor bus):
- planner: observation -> one B-spline segment (knots in data frames
  relative to the observation). A new segment is planned when the
  current one is --replan-frac used.
- control (--hz): samples the active segment by wall-clock time,
  tau = elapsed_s * fps * speed, so the speed does not depend on the
  inference rate. --speed 2 runs the curve twice as fast as the demos.

A new segment starts at the point near "now" that is closest to the
last command (follows the official executor). --blend fades from the
old to the new segment (smoothstep) so the switch does not jump. Each
joint step is limited to --max-step deg per tick.

Per episode: press Enter when the cube is in the bin (or the episode
ends at --duration), then type s (success) or f (fail). Results go to
outputs/portfolio/results_bsp.csv; front video + log to bsp/runs/.

--full-record: full-rate front and wrist clips (30 fps, H.264) plus
ep<n>_showcase.mp4 (side by side, caption and timer) for showcases.
--showcase-cam adds a side view from a camera the policy does not use.
--preview (with --full-record) shows the side and front views live,
also before Enter, so the cameras can be aimed.

--policy act_chunk runs the plain ACT (no B-spline) through the same
executor: its action chunk (one action per demo frame) is a straight
line from frame to frame, so --speed scales it the same way.

Usage:
  python outputs/portfolio/bsp/run_bsp.py --policy act --speed 1 \
    --episodes 5 --spot X1

Speed sweep with debug logs (one folder, a diagnosis table at the end):
  python outputs/portfolio/bsp/run_bsp.py --policy diffusion --spot SQ \
    --sweep 1,1.25,1.5,1.75,2,2.5,3 --episodes 1
--debug (on by default in a sweep) saves ep<n>_debug.npz per episode;
diagnose_bsp.py reads them.
"""

import argparse
import csv
import json
import select
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

sys.path.insert(0, str(Path(__file__).parent))
from bspline import make_spline, segment_range  # noqa: E402
from diagnose_bsp import print_table, summarize  # noqa: E402
from segment import predict_segment  # noqa: E402
from video_record import ExtraCamera, Preview, Recorder, finish  # noqa: E402

HERE = Path(__file__).parent
TRAIN = HERE.parent.parent / "train"
RUNS = {"act": TRAIN / "act_bsp_ws30k/checkpoints/last/pretrained_model",
        "diffusion": TRAIN / "diffusion_bsp_ws40k/checkpoints/last/"
        "pretrained_model",
        "act_chunk": TRAIN / "act_all_ft50k/checkpoints/050000/"
        "pretrained_model"}
LABELS = {"act": "ACT-BSP", "diffusion": "DIFFUSION-BSP",
          "act_chunk": "ACT-CHUNK"}
DATA_FPS = 30  # third-act-attempt; plain policies have no bsp.json
RESULTS = HERE.parent / "results_bsp.csv"
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034620-if00"
TASK = "pick green cube and place into the bin"
DONE_TEXT = "the cube is in the bin"


class LinearChunk:
    """Action chunk (rows = demo frames from the observation) as a curve.

    Straight line between consecutive actions; valid on [0, rows - 1].
    Same call style as a BSpline: s scalar -> (dim,), s (n,) -> (n, dim).
    """

    def __init__(self, actions):
        self.a = np.asarray(actions, dtype=np.float64)

    def __call__(self, s):
        s = np.asarray(s, dtype=np.float64)
        i = np.clip(np.floor(s).astype(int), 0, len(self.a) - 2)
        f = (s - i)[..., None]
        return self.a[i] * (1 - f) + self.a[i + 1] * f


def make_curve(out, degree):
    """(curve, t0, t1) from one policy output; degree None = chunk."""
    if degree is None:
        return LinearChunk(out), 0.0, float(len(out) - 1)
    t0, t1 = segment_range(out, degree)
    return make_spline(out, degree), t0, t1


def make_robot(args):
    cams = {name: OpenCVCameraConfig(index_or_path=dev, width=640,
                                     height=480, fps=30)
            for name, dev in (("wrist", args.wrist), ("front", args.front))}
    cfg = SO101FollowerConfig(port=PORT, id="my_follower_arm", cameras=cams)
    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


class Executor:
    def __init__(self, robot, policy, pre, post, keys, args, degree):
        self.robot, self.policy, self.pre, self.post = robot, policy, pre, post
        self.keys, self.args, self.degree = keys, args, degree
        self.fps = float(args.fps)
        self.bus = threading.Lock()
        self.seg_lock = threading.Lock()
        self.device = torch.device("cuda")

    def tau(self):
        """Episode time in demo frames (speed-scaled)."""
        return (time.perf_counter() - self.t_start) * self.fps \
            * self.args.speed

    def observe(self):
        with self.bus:
            obs = self.robot.get_observation()
        state = np.array([obs[k] for k in self.keys], np.float32)
        frame = {"observation.state": state,
                 "observation.images.front": obs["front"],
                 "observation.images.wrist": obs["wrist"]}
        return obs, state, frame

    def plan(self, frame, tau_obs, state):
        t = time.perf_counter()
        batch = prepare_observation_for_inference(
            dict(frame), self.device, TASK, "so101_follower")
        with torch.inference_mode():
            params = self.post(predict_segment(self.policy,
                                               self.pre(batch), self.noise))
        params = params[0].float().cpu().numpy()
        self.latency.append(time.perf_counter() - t)
        spl, t0, t1 = make_curve(params, self.degree)
        now = self.tau()
        natural = now - tau_obs
        grid = np.clip(np.arange(natural - 3, natural + 6, 0.25),
                       t0, t1 - 1e-6)
        last = self.last_cmd
        if last is not None:
            err = np.abs(spl(grid) - last).sum(1)
            start = float(grid[int(np.nanargmin(err))])
        else:
            start = float(np.clip(natural, t0, t1 - 1e-6))
        with self.seg_lock:
            old = self.seg
            self.seg = (spl, start, now, t0, t1, self.n_segments)
            # margin: demo frames left on the old segment at the switch
            margin = np.nan
            if old is not None:
                _, s0, n0, _, e1, _ = old
                margin = e1 - (s0 + now - n0)
            # the fade must end before the old segment does
            left = margin / (self.fps * self.args.speed)
            fade = min(self.args.blend, left) if old is not None else 0
            t_now = time.perf_counter()
            self.fade = (old, t_now, t_now + fade) if fade > 0 else None
        self.n_segments += 1
        if self.args.debug:
            # arm joints only: a closed gripper on the cube always lags
            seam = lag = np.nan
            if last is not None:
                seam = float(np.abs(spl(start) - last)[:5].max())
                lag = float(np.abs(state - last)[:5].max())
            self.plans.append([time.perf_counter() - self.t_start,
                               tau_obs, self.latency[-1], natural, start,
                               t0, t1, margin, seam, lag])

    def progress(self):
        with self.seg_lock:
            if self.seg is None:
                return 1.0
            _, start, now0, t0, t1, _ = self.seg
        return (start + self.tau() - now0 - t0) / max(t1 - t0, 1e-6)

    def control(self):
        period = 1.0 / self.args.hz
        nxt = time.perf_counter()
        while self.running:
            with self.seg_lock:
                seg, fade = self.seg, self.fade
            if seg is not None:
                spl, start, now0, t0, t1, sid = seg
                s_raw = start + self.tau() - now0
                s = float(np.clip(s_raw, t0, t1 - 1e-6))
                target = spl(s)
                t_now = time.perf_counter()
                if fade is not None and t_now < fade[2]:
                    o_spl, o_start, o_now, o_t0, o_t1, _ = fade[0]
                    o_s = np.clip(o_start + self.tau() - o_now, o_t0,
                                  o_t1 - 1e-6)
                    a = (t_now - fade[1]) / (fade[2] - fade[1])
                    a = a * a * (3 - 2 * a)  # smoothstep
                    target = (1 - a) * o_spl(o_s) + a * target
                raw = target
                if self.last_cmd is not None:
                    step = np.clip(target - self.last_cmd,
                                   -self.args.max_step, self.args.max_step)
                    target = self.last_cmd + step
                if np.all(np.isfinite(target)):
                    self.last_cmd = target
                    t_wait = time.perf_counter()
                    with self.bus:
                        wait = time.perf_counter() - t_wait
                        self.robot.send_action(
                            dict(zip(self.keys, map(float, target))))
                    now_s = time.perf_counter() - self.t_start
                    self.log.append([now_s, self.tau(), *target])
                    if self.args.debug:
                        self.ticks.append([now_s, self.tau(), sid, s_raw,
                                           t1, wait, *raw, *target])
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def done_pressed(self):
        """True once Enter is pressed in the terminal."""
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
        return False

    def episode(self):
        self.policy.reset()
        self.seg, self.last_cmd, self.fade = None, None, None
        # one start noise per episode: plans differ only by observation
        self.noise = None
        cfg = self.policy.config
        if self.args.fixed_noise and cfg.type == "diffusion":
            self.noise = torch.randn(
                (1, cfg.horizon, cfg.action_feature.shape[0]),
                device=self.device)
        self.latency, self.log, self.frames = [], [], []
        self.ticks, self.plans = [], []
        self.n_segments, self.running = 0, True
        self.t_start = time.perf_counter()
        ctl = threading.Thread(target=self.control, daemon=True)
        ctl.start()
        done_s = None
        print(f"running... press Enter when {DONE_TEXT}")
        while time.perf_counter() - self.t_start < self.args.duration:
            if self.done_pressed():
                done_s = time.perf_counter() - self.t_start
                break
            if self.progress() >= self.args.replan_frac:
                tau_obs = self.tau()
                obs, state, frame = self.observe()
                self.frames.append((time.perf_counter() - self.t_start,
                                    obs["front"]))
                self.plan(frame, tau_obs, state)
            else:
                time.sleep(0.005)
        self.running = False
        ctl.join()
        return done_s


def go_to(ex, pose, seconds=2.5):
    """Move to a joint pose by linear interpolation."""
    with ex.bus:
        obs = ex.robot.get_observation()
    start = np.array([obs[k] for k in ex.keys])
    n = int(seconds * 50)
    for i in range(1, n + 1):
        q = start + (pose - start) * i / n
        with ex.bus:
            ex.robot.send_action(dict(zip(ex.keys, map(float, q))))
        time.sleep(1 / 50)


def save_episode(ex, out, name):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}_cmd.npy", np.array(ex.log))
    if ex.args.debug:
        a = ex.args
        np.savez(out / f"{name}_debug.npz", ticks=np.array(ex.ticks),
                 plans=np.array(ex.plans), speed=a.speed, hz=a.hz,
                 max_step=a.max_step, replan_frac=a.replan_frac,
                 fps=a.fps, ddim_steps=a.ddim_steps or -1, blend=a.blend,
                 fixed_noise=a.fixed_noise)
    if ex.frames:
        h, w = ex.frames[0][1].shape[:2]
        vw = cv2.VideoWriter(str(out / f"{name}_front.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        times = np.array([t for t, _ in ex.frames])
        for t in np.arange(0, times[-1], 1 / 15):
            k = int(np.searchsorted(times, t, side="right")) - 1
            vw.write(cv2.cvtColor(ex.frames[max(k, 0)][1],
                                  cv2.COLOR_RGB2BGR))
        vw.release()


def safe_read(cam):
    try:
        return cam.async_read(timeout_ms=500)
    except (TimeoutError, RuntimeError):
        return None


def make_recorders(robot, extra, out, name):
    recs = []
    if extra is not None:
        recs.append(Recorder("side", extra.read, out / f"{name}_side.avi"))
    for cam in ("front", "wrist"):
        c = robot.cameras[cam]
        recs.append(Recorder(f"{cam}_full", lambda c=c: safe_read(c),
                             out / f"{name}_{cam}_full.avi"))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(RUNS), required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--spot", default="X1")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--replan-frac", type=float, default=0.5)
    ap.add_argument("--max-step", type=float, default=4.0,
                    help="max joint change per control tick (deg)")
    ap.add_argument("--front", default="/dev/video6")
    ap.add_argument("--wrist", default="/dev/video4")
    ap.add_argument("--sweep", default="",
                    help="comma list of speeds, e.g. 1,1.5,2; runs "
                    "--episodes per speed with --debug on")
    ap.add_argument("--debug", action="store_true",
                    help="save per-tick and per-plan logs")
    ap.add_argument("--ddim-steps", type=int, default=None,
                    help="diffusion only: override denoising steps")
    ap.add_argument("--blend", type=float, default=0.0,
                    help="fade from the old to the new segment (s); "
                    "capped at the time left on the old segment")
    ap.add_argument("--full-record", action="store_true",
                    help="also record full-rate front/wrist videos and a "
                    "side-by-side showcase clip")
    ap.add_argument("--showcase-cam", default=None,
                    help="with --full-record: extra side-view camera, e.g. "
                    "/dev/video8 (not used by the policy)")
    ap.add_argument("--preview", action="store_true",
                    help="with --full-record: live window of the side "
                    "and front views while recording")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="diffusion only: reuse one start noise for every "
                    "plan in an episode (new noise each episode)")
    args = ap.parse_args()
    speeds = ([float(s) for s in args.sweep.split(",")] if args.sweep
              else [args.speed])
    args.debug = args.debug or bool(args.sweep)

    path = RUNS[args.policy]
    meta = (json.loads((path / "bsp.json").read_text())
            if (path / "bsp.json").exists()
            else {"fps": DATA_FPS, "degree": None})
    args.fps = meta["fps"]
    cfg = PreTrainedConfig.from_pretrained(path)
    policy = get_policy_class(cfg.type).from_pretrained(path).to("cuda")
    policy.eval()
    if args.ddim_steps and cfg.type == "diffusion":
        policy.diffusion.num_inference_steps = args.ddim_steps
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path))
    robot = make_robot(args)
    extra = (ExtraCamera(args.showcase_cam)
             if args.showcase_cam and args.full_record else None)
    preview = None
    if args.preview and args.full_record:
        front = robot.cameras["front"]
        idle = {"side": extra.read} if extra is not None else {}
        idle["front_full"] = lambda: safe_read(front)
        preview = Preview(list(idle), idle)
    keys = list(robot.action_features)
    ex = Executor(robot, policy, pre, post, keys, args, meta["degree"])
    with ex.bus:
        obs = robot.get_observation()
    home = np.array([obs[k] for k in keys])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = HERE / "runs" / f"sweep_{args.policy}_{args.spot}_{stamp}"
    new = not RESULTS.exists()
    summaries = []
    label = LABELS[args.policy]
    name = label.lower().replace("-", "_")
    try:
        for speed in speeds:
            args.speed = speed
            if args.sweep:
                run = f"{sweep_dir.name}/x{speed:g}"
            else:
                run = f"{name}_{args.spot}_x{speed:g}_{stamp}"
            out = HERE / "runs" / run
            for ep in range(args.episodes):
                input(f"\n[{run}] episode {ep + 1}/{args.episodes}: reset "
                      f"the cube, then press Enter to start ")
                recs = (make_recorders(robot, extra, out, f"ep{ep}")
                        if args.full_record else [])
                t_rec = time.perf_counter()
                for r in recs:
                    r.start(t_rec)
                if preview is not None:
                    preview.show(recs)
                done_s = ex.episode()
                go_to(ex, home)
                for r in recs:
                    r.stop()
                res = ""
                while res not in ("s", "f"):
                    res = input("result: s = success, f = fail: ").strip()
                lat = 1000 * float(np.mean(ex.latency)) if ex.latency else 0
                save_episode(ex, out, f"ep{ep}")
                if recs:
                    verdict = "success" if res == "s" else "fail"
                    finish(recs, out, f"ep{ep}",
                           f"{label}  {speed:g}x  real time  {verdict}")
                row = {"policy": name, "spot": args.spot,
                       "speed": speed, "trial": ep + 1,
                       "success": int(res == "s"),
                       "done_s": "" if done_s is None or res != "s"
                       else round(done_s, 1),
                       "plan_ms": round(lat), "segments": ex.n_segments,
                       "run": run}
                with open(RESULTS, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                        new = False
                    w.writerow(row)
                print(f"saved: success={row['success']} "
                      f"done={row['done_s']}s plan={row['plan_ms']}ms "
                      f"segments={ex.n_segments}")
                if args.debug:
                    s = summarize(out / f"ep{ep}_debug.npz")
                    s.update(success=row["success"], done_s=row["done_s"],
                             file=f"{run}/ep{ep}")
                    summaries.append(s)
                    print_table([s])
    finally:
        ex.running = False
        robot.disconnect()
        if preview is not None:
            preview.close()
        if extra is not None:
            extra.release()
        if args.sweep and summaries:
            sweep_dir.mkdir(parents=True, exist_ok=True)
            (sweep_dir / "summary.json").write_text(
                json.dumps(summaries, indent=1))
            print_table(summaries, hints=True)
            print(f"\nsummary: {sweep_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
