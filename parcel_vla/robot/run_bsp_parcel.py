"""Run Diffusion-BSP or SmolVLA-BSP on the parcel task (BSP executor).

Uses the executor of run_bsp.py (planner thread + --hz control
thread, wall-clock curve time) with the parcel setup:
- cameras top (RealSense colour, 848x480), wrist and side (640x480);
- the parcel task text; images converted on the GPU;
- --model diffusion_bsp (default): diffusion_bsp_parcel_40k,
  patch_diffusion() (mixed camera sizes), DDIM sampling,
  --ddim-steps (default 10, about 0.1 s per plan);
- --model smolvla_bsp: smolvla_bsp_parcel_20k; its saved processors
  rename the cameras and tokenize the task text.
- --model diffusion_bsp_b / smolvla_bsp_b: the same, trained on
  dataset B (diffusion_bsp_parcel_b_40k, smolvla_bsp_parcel_b_40k).
- --model smolvla_bsp_b_dg: smolvla_bsp_b + 15k steps on dataset B and
  the DAgger corrections (smolvla_bsp_parcel_b_dagger_15k).
- --model smolvla_bsp_b_dgt: the same, with the idle home tail of each
  correction cut (smolvla_bsp_parcel_b_dagger_trim_15k).
- --policy-path <folder>: run this checkpoint folder (for example a
  model from the Hub) with the settings of --model.

Per episode: press Enter or Right arrow to start, and again when the
parcel is off the table (or the episode ends at --duration). These two
keys work with any window in focus (global listener). The robot goes
back to its pose at launch. Then, in the terminal, type s (success) or
f (fail) and, for a fail, the phase: p (push), f (flip), e (exit).

Results: runs/results_bsp_parcel.csv (one row per episode, with
the --level and log-only fields of eval_parcel.sh). Per episode in
runs/bsp/<model>_parcel_<level>_<time>/: ep<n>_cams.mp4 (all 3
cameras, 30 fps, from start to done), ep<n>_front.mp4 (top camera at
the planning frames only) and ep<n>_cmd.npy (command log).

Usage:
  python -m parcel_vla.robot.run_bsp_parcel --level easy \
    --episodes 3 --parcels 3 --where near_flip
"""

import argparse
import contextlib
import csv
import json
import os
import sys
import termios
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from parcel_vla import paths
from parcel_vla.inference_improvement import cpp_control
from parcel_vla.inference_improvement import fast_denoise
from parcel_vla.robot import run_bsp as rb
from parcel_vla.robot.rollout_mixed import fast_prepare, start_preview
from parcel_vla.training.train_mixed import patch_diffusion

from lerobot.cameras.opencv.configuration_opencv import (
    OpenCVCameraConfig)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import (
    get_policy_class, make_pre_post_processors)
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import (
    SO101FollowerConfig)
from lerobot.utils.keyboard_input import create_key_listener

RUNS = {"diffusion_bsp": ("diffusion_bsp_parcel_40k", "Diffusion-BSP"),
        "smolvla_bsp": ("smolvla_bsp_parcel_20k", "SmolVLA-BSP"),
        "diffusion_bsp_b": ("diffusion_bsp_parcel_b_40k",
                            "Diffusion-BSP (dataset B)"),
        "smolvla_bsp_b": ("smolvla_bsp_parcel_b_40k",
                          "SmolVLA-BSP (dataset B)"),
        "smolvla_bsp_b_dg": ("smolvla_bsp_parcel_b_dagger_15k",
                             "SmolVLA-BSP (dataset B + DAgger 1)"),
        "smolvla_bsp_b_dgt": ("smolvla_bsp_parcel_b_dagger_trim_15k",
                              "SmolVLA-BSP (B + DAgger 1, tails cut)")}
RESULTS = paths.RUNS / "results_bsp_parcel.csv"
WRIST = paths.WRIST_CAM
SIDE = paths.SIDE_CAM
CAMS = ("top", "wrist", "side")

rb.TASK = ("sort the parcel so the label faces down and push it off the "
           "table")
rb.DONE_TEXT = "the parcel is off the table"
rb.prepare_observation_for_inference = fast_prepare


def realsense_colour():
    """The RealSense video node with YUYV colour (as record_parcel.sh)."""
    if paths.TOP_CAM:
        return paths.TOP_CAM
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        if "RealSense" not in (d / "name").read_text():
            continue
        dev = f"/dev/{d.name}"
        fmts = os.popen(f"v4l2-ctl -d {dev} --list-formats 2>/dev/null")
        if "YUYV" in fmts.read():
            return dev
    raise SystemExit("RealSense colour node not found")


def make_robot():
    cams = {
        "top": OpenCVCameraConfig(index_or_path=realsense_colour(),
                                  width=848, height=480, fps=30,
                                  fourcc="YUYV"),
        "wrist": OpenCVCameraConfig(index_or_path=WRIST, width=640,
                                    height=480, fps=30, fourcc="MJPG"),
        "side": OpenCVCameraConfig(index_or_path=SIDE, width=640,
                                   height=480, fps=30, fourcc="MJPG"),
    }
    cfg = SO101FollowerConfig(port=rb.PORT, id="my_follower_arm",
                              cameras=cams, num_read_retries=10)
    robot = make_robot_from_config(cfg)
    try:
        robot.connect()
    except ConnectionError as err:
        # configure() writes with no retry; one bad status packet stops
        # connect() after the bus and cameras are open: redo only that
        for attempt in range(1, 4):
            print(f"motor write failed ({err}); retry {attempt}/3")
            time.sleep(0.5)
            try:
                robot.configure()
                break
            except ConnectionError as e:
                err = e
        else:
            raise
    return robot


class ParcelExecutor(rb.Executor):
    """run_bsp Executor with the parcel cameras and a global done key.

    Right arrow or Enter ends the episode with any window in focus (the
    preview window can hold the focus). Enter in the terminal also works.
    """

    key_done = False
    running = False

    def start_keys(self):
        def on_key(name):
            if name in ("right", "enter"):
                self.key_done = True

        self.listener = create_key_listener(
            on_key, controls_help="Right arrow / Enter = start / done")

    def wait_start(self, prompt):
        """Wait for Enter in the terminal or Right arrow / Enter anywhere."""
        flush_stdin()
        print(prompt, end="", flush=True)
        self.key_done = False
        while not self.done_pressed():
            time.sleep(0.02)
        print()
        time.sleep(0.3)  # key repeat of this press must not end the run
        flush_stdin()
        self.key_done = False

    def done_pressed(self):
        if self.key_done:
            self.key_done = False
            return True
        return super().done_pressed()

    def observe(self):
        with self.bus:
            obs = self.robot.get_observation()
        state = np.array([obs[k] for k in self.keys], np.float32)
        frame = {"observation.state": state}
        for c in CAMS:
            frame[f"observation.images.{c}"] = obs[c]
        obs["front"] = obs["top"]  # run_bsp keeps "front" frames for video
        return obs, state, frame


class EpisodeVideo(threading.Thread):
    """All cameras in one 30 fps mp4 (top large, wrist and side right).

    Copies the newest frame of each camera (does not consume frames),
    so the planner and the control thread are not slowed.
    """

    def __init__(self, cameras, path, caption, fps=30, width=1280):
        super().__init__(daemon=True)
        self.cams, self.path, self.caption = cameras, path, caption
        self.fps = fps
        self.big_w = width * 2 // 3 // 2 * 2
        self.big_h = self.big_w * 9 // 16 // 2 * 2
        self.small_w = (width - self.big_w) // 2 * 2
        self.small_h = self.small_w * 3 // 4 // 2 * 2
        self.w = self.big_w + self.small_w
        self.h = max(self.big_h, 2 * self.small_h)
        self.running = True

    def _frame(self, name, w, h):
        cam = self.cams[name]
        with cam.frame_lock:
            f = cam.latest_frame
        if f is None:
            return np.zeros((h, w, 3), np.uint8)
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

    def run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(self.path),
                             cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                             (self.w, self.h))
        t0 = time.perf_counter()
        n = 0
        while self.running:
            canvas = np.zeros((self.h, self.w, 3), np.uint8)
            canvas[:self.big_h, :self.big_w] = self._frame(
                "top", self.big_w, self.big_h)
            for i, name in enumerate(("wrist", "side")):
                y = i * self.small_h
                canvas[y:y + self.small_h, self.big_w:] = self._frame(
                    name, self.small_w, self.small_h)
            t = time.perf_counter() - t0
            cv2.putText(canvas, f"{self.caption}  {t:5.1f} s",
                        (10, self.h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)
            vw.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            n += 1
            time.sleep(max(0.0, t0 + n / self.fps - time.perf_counter()))
        vw.release()

    def stop(self):
        self.running = False
        self.join()


def warm_up(ex, n=3):
    """Plan n times with the robot still: the first plan compiles."""
    _, _, frame = ex.observe()
    batch = rb.prepare_observation_for_inference(
        dict(frame), ex.device, rb.TASK, "so101_follower")
    for i in range(n):
        t = time.perf_counter()
        with torch.inference_mode():
            rb.predict_segment(ex.policy, ex.pre(batch), None)
        torch.cuda.synchronize()
        print(f"warm-up plan {i + 1}/{n}: "
              f"{time.perf_counter() - t:.2f} s")


def flush_stdin():
    """Drop keys typed while the robot ran (stray Enter, arrows)."""
    with contextlib.suppress(termios.error, OSError):
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def ask(prompt, choices):
    flush_stdin()
    res = ""
    while res not in choices:
        res = input(prompt).strip()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RUNS), default="diffusion_bsp")
    ap.add_argument("--level", choices=["easy", "medium", "hard"],
                    required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--replan-frac", type=float, default=0.5)
    ap.add_argument("--max-step", type=float, default=4.0,
                    help="max joint change per control tick (deg)")
    ap.add_argument("--blend", type=float, default=0.0)
    ap.add_argument("--ddim-steps", type=int, default=10)
    ap.add_argument("--flow-steps", type=int, default=10,
                    help="smolvla_bsp: flow matching steps per plan "
                    "(10 = training default; 5 halves the plan time)")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="one start noise per episode (smaller jumps "
                    "between segments)")
    ap.add_argument("--compile", default="off",
                    choices=["off", "default", "reduce-overhead"],
                    help="smolvla_bsp: compiled denoise step "
                    "(fast_denoise.py); default ~3.4 ms/step, "
                    "reduce-overhead (CUDA graph) ~3.1, off ~13")
    ap.add_argument("--trt", default="off",
                    choices=["off", "fp16", "fp32", "bf16"],
                    help="dataset B models: TensorRT engines of "
                    "trt_plan.py (build them first)")
    ap.add_argument("--control", default="python",
                    choices=["python", "cpp"],
                    help="cpp: control loop and servo bus in "
                    "ctl/bsp_ctl (cpp_control.py)")
    ap.add_argument("--policy-path", default=None,
                    help="checkpoint folder to run with the settings of "
                    "--model (default: the --model run in TRAIN_DIR)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-video", action="store_true",
                    help="no ep<n>_cams.mp4 (3 cameras, 30 fps)")
    for f in ("parcels", "parcel", "where", "note"):
        ap.add_argument(f"--{f}", default="", help="log only")
    ap.add_argument("--label", default="mixed", help="log only")
    args = ap.parse_args()

    ckpt, name = RUNS[args.model]
    policy_dir = rb.TRAIN / ckpt / "checkpoints/last/pretrained_model"
    if args.policy_path:
        policy_dir = Path(args.policy_path)
    meta = json.loads((policy_dir / "bsp.json").read_text())
    args.fps = meta["fps"]
    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    settings = f"x{args.speed:g}; max step {args.max_step:g}"
    settings = f"{settings}; blend {args.blend:g}"
    if cfg.type == "smolvla":
        cfg.num_steps = args.flow_steps
        settings = f"flow {args.flow_steps}; {settings}"
    if cfg.type == "diffusion":
        patch_diffusion()
        cfg.noise_scheduler_type = "DDIM"
        cfg.num_inference_steps = args.ddim_steps
        settings = f"DDIM {args.ddim_steps}; {settings}"
    policy = get_policy_class(cfg.type).from_pretrained(policy_dir,
                                                        config=cfg)
    if cfg.type == "smolvla":
        # train_bsp.py saved 146M VLM weights as float32 (lerobot-train
        # keeps the VLM in bf16): bf16 plans in ~0.21 s, not ~0.35 s
        vlm = policy.model.vlm_with_expert.vlm
        for prm in vlm.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)
        if args.fixed_noise:
            settings += "; fixed noise"
        if args.compile != "off":
            fast_denoise.enable(policy.model, args.compile)
            settings += f"; compile {args.compile}"
    policy.to("cuda").eval()
    if args.trt != "off":
        from parcel_vla.inference_improvement import trt_plan

        if args.model not in trt_plan.CKPTS:
            raise SystemExit(f"--trt: no engines for {args.model}")
        trt_plan.enable(policy, args.model, args.trt)
        settings += f"; trt {args.trt}"
    pre, post = make_pre_post_processors(cfg,
                                         pretrained_path=str(policy_dir))

    robot = make_robot()
    if not args.no_preview:
        start_preview(robot, f"{args.model} {args.level}")
    keys = list(robot.action_features)
    cls, ctl = ParcelExecutor, None
    if args.control == "cpp":
        # LeRobot configured the motors in connect(); bsp_ctl now owns
        # the port until the end of the run
        robot.bus.port_handler.closePort()
        ctl = cpp_control.CppControl(hz=args.hz, port=rb.PORT)
        cls = type("CppParcelExecutor",
                   (cpp_control.CppExecutorMixin, ParcelExecutor), {})
        settings += "; control cpp"
    ex = cls(robot, policy, pre, post, keys, args, meta["degree"])
    ex.cpp = ctl
    ex.start_keys()
    if ctl is not None:
        t = time.perf_counter()
        while ctl.state()[0] is None and time.perf_counter() - t < 2:
            time.sleep(0.02)
        state = ctl.state()[0]
        if state is None:
            ctl.stop()
            raise SystemExit("bsp_ctl: no arm state from the bus")
        home = np.array(state)
    else:
        with ex.bus:
            obs = robot.get_observation()
        home = np.array([obs[k] for k in keys])
    go_to = cpp_control.go_to_cpp if ctl is not None else rb.go_to
    if args.compile != "off" or args.trt != "off":
        warm_up(ex)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = f"{args.model}_parcel_{args.level}_{stamp}"
    out = paths.RUNS / "bsp" / run
    new = not RESULTS.exists()
    print(f"{name} ({ckpt}/last, {settings}) level={args.level} -> {out}")
    try:
        for ep in range(args.episodes):
            ex.wait_start(f"\n[{args.level}] episode {ep + 1}/"
                          f"{args.episodes}: robot at home, parcel ready? "
                          "Enter or Right arrow to start ")
            video = None
            if not args.no_video:
                video = EpisodeVideo(
                    robot.cameras, out / f"ep{ep}_cams.mp4",
                    f"{name} x{args.speed:g} {args.level} ep{ep}")
                video.start()
            done_s = ex.episode()
            ex.key_done = False
            if video is not None:
                video.stop()
            go_to(ex, home)
            res = ask("result: s = success, f = fail: ", ("s", "f"))
            phase = ""
            if res == "f":
                phase = {"p": "push", "f": "flip", "e": "exit"}[ask(
                    "failed in: p = push, f = flip, e = exit: ",
                    ("p", "f", "e"))]
            lat = 1000 * float(np.mean(ex.latency)) if ex.latency else 0
            rb.save_episode(ex, out, f"ep{ep}")
            if ctl is not None:
                np.save(out / f"ep{ep}_cpp_ticks.npy", ex.cpp_ticks)
            row = {"time": time.strftime("%F_%T"),
                   "model": args.model,
                   "checkpoint": f"{ckpt}/last",
                   "settings": settings,
                   "level": args.level, "episode": ep,
                   "success": int(res == "s"), "fail_phase": phase,
                   "done_s": "" if done_s is None or res != "s"
                   else round(done_s, 1),
                   "plan_ms": round(lat), "segments": ex.n_segments,
                   "parcels": args.parcels, "kind": args.parcel,
                   "where": args.where, "label": args.label,
                   "note": args.note, "run": run}
            with open(RESULTS, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                    new = False
                w.writerow(row)
            print(f"saved: success={row['success']} {phase} "
                  f"done={row['done_s']}s plan={row['plan_ms']}ms "
                  f"segments={ex.n_segments}")
    finally:
        ex.running = False
        if ctl is not None:
            ctl.stop()
            robot.bus.port_handler.openPort()
        robot.disconnect()
    if RESULTS.exists():
        print(f"\n{RESULTS}:")
        print(RESULTS.read_text())


if __name__ == "__main__":
    main()
