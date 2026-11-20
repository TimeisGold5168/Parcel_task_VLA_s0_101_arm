"""Offline benchmarks of the control path (items 11 and 12, no motion).

  bus   servo bus read time: LeRobot sync_read (Python) against bsp_ctl
        (C++). Read only: Present_Position, no write, no torque change.
  cams  the real cameras: read_latest() time and frame age at the plan
        (newest frame time to the read), 3 cameras as the parcel task.
  loop  tick timing of the BSP executor while the planner runs on the
        GPU. A fake arm takes the commands (no bus); the planner gets
        fixed dataset B frames (or the real cameras with --cams real).
        --control python | cpp, --video on (the 3-camera mp4 thread of
        run_bsp_parcel.py).

  python outputs/portfolio/bsp/bench_control.py bus
  python outputs/portfolio/bsp/bench_control.py cams
  python outputs/portfolio/bsp/bench_control.py loop --control python \
    --video on --json bench_control.jsonl
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "parcel_task"))
import cpp_control  # noqa: E402

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034620-if00"
KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def stats(x):
    x = np.asarray(x, np.float64)
    return {"p50": round(float(np.median(x)), 3),
            "p95": round(float(np.percentile(x, 95)), 3),
            "max": round(float(x.max()), 3), "n": int(len(x))}


def emit(row, path):
    print(json.dumps(row))
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- bus

def bench_bus(args):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal = json.loads(cpp_control.CALIB.read_text())
    motors = {n: Motor(c["id"], "sts3215",
                       MotorNormMode.RANGE_0_100 if n == "gripper"
                       else MotorNormMode.DEGREES) for n, c in cal.items()}
    calib = {n: MotorCalibration(**c) for n, c in cal.items()}
    bus = FeetechMotorsBus(PORT, motors, calibration=calib)
    bus.connect()
    ts, q_py = [], None
    for _ in range(args.n):
        t = time.perf_counter()
        q_py = bus.sync_read("Present_Position")
        ts.append((time.perf_counter() - t) * 1e3)
        time.sleep(0.02)
    bus.port_handler.closePort()
    emit({"bench": "bus_read", "impl": "python_lerobot", **stats(ts)},
         args.json)

    ctl = cpp_control.CppControl(hz=50, port=PORT)  # HOLD: reads only
    time.sleep(args.n * 0.02)
    state, _ = ctl.state()
    t = ctl.ticks()
    errs = int(ctl._out("read_errors"))
    ctl.stop()
    emit({"bench": "bus_read", "impl": "cpp_bsp_ctl", "errors": errs,
          **stats(t[1:, 3])}, args.json)
    if state is not None and q_py is not None:
        d = np.abs(np.array([q_py[k.removesuffix(".pos")] for k in KEYS])
                   - state)
        print(f"Python vs C++ arm state: max |diff| {d.max():.2f} "
              "(deg / gripper %; the arm can move a little)")


# ---------------------------------------------------------------- cams

def open_cams():
    from run_bsp_parcel import SIDE, WRIST, realsense_colour

    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import (
        OpenCVCameraConfig)

    cfgs = {"top": OpenCVCameraConfig(index_or_path=realsense_colour(),
                                      width=848, height=480, fps=30,
                                      fourcc="YUYV"),
            "wrist": OpenCVCameraConfig(index_or_path=WRIST, width=640,
                                        height=480, fps=30, fourcc="MJPG"),
            "side": OpenCVCameraConfig(index_or_path=SIDE, width=640,
                                       height=480, fps=30, fourcc="MJPG")}
    cams = {n: OpenCVCamera(c) for n, c in cfgs.items()}
    for c in cams.values():
        c.connect()
        c.async_read(timeout_ms=2000)
    return cams


def bench_cams(args):
    cams = open_cams()
    read, age = {n: [] for n in cams}, {n: [] for n in cams}
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        for n, c in cams.items():
            t = time.perf_counter()
            c.read_latest()
            now = time.perf_counter()
            read[n].append((now - t) * 1e3)
            age[n].append((now - c.latest_timestamp) * 1e3)
        time.sleep(0.1)
    for c in cams.values():
        c.disconnect()
    for n in cams:
        emit({"bench": "camera", "cam": n, "read_ms": stats(read[n]),
              "age_ms": stats(age[n])}, args.json)


# ---------------------------------------------------------------- loop

class FakeCam:
    def __init__(self, frame):
        self.frame_lock = threading.Lock()
        self.latest_frame = frame
        self.latest_timestamp = time.perf_counter()

    def read_latest(self, max_age_ms=500):
        return self.latest_frame


class FakeRobot:
    """Fixed observation; send_action only counts (no bus)."""

    def __init__(self, frames, state, cams=None):
        self.cameras = cams or {n: FakeCam(f) for n, f in frames.items()}
        self.state = state
        self.sent = 0

    def get_observation(self):
        obs = dict(zip(KEYS, map(float, self.state)))
        for n, c in self.cameras.items():
            obs[n] = c.read_latest()
        return obs

    def send_action(self, action):
        self.sent += 1
        return action


def dataset_frame(frame_idx=30000):
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path.home() / ".cache/huggingface/lerobot/Low101/parcel-sort-b"
    ds = LeRobotDataset("Low101/parcel-sort-b", root=root)
    item = ds[frame_idx]
    frames = {}
    for n in ("top", "wrist", "side"):
        img = item[f"observation.images.{n}"]  # (3, H, W) float 0-1
        frames[n] = (img.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return frames, item["observation.state"].numpy()


def bench_loop(args):
    import run_bsp_parcel as rbp
    import trt_plan

    from lerobot.policies.factory import make_pre_post_processors

    policy, path = trt_plan.load_policy(args.model)
    if args.trt != "off":
        trt_plan.enable(policy, args.model, args.trt)
    if args.compile != "off":
        import fast_denoise

        fast_denoise.enable(policy.model, args.compile)
    pre, post = make_pre_post_processors(policy.config,
                                         pretrained_path=str(path))
    meta = json.loads((path / "bsp.json").read_text())
    frames, state = dataset_frame()
    cams = open_cams() if args.cams == "real" else None
    robot = FakeRobot(frames, state, cams)
    a = SimpleNamespace(speed=args.speed, hz=args.hz, replan_frac=0.5,
                        max_step=4.0, blend=0.2, fixed_noise=True,
                        debug=True, fps=meta["fps"], duration=args.seconds,
                        ddim_steps=10)
    cls, ctl = rbp.ParcelExecutor, None
    if args.control == "cpp":
        ctl = cpp_control.CppControl(hz=args.hz, port=None)
        cls = type("CppBench", (cpp_control.CppExecutorMixin,
                                rbp.ParcelExecutor), {})
    ex = cls(robot, policy, pre, post, KEYS, a, meta["degree"])
    ex.cpp = ctl
    ex.done_pressed = lambda: False
    if ctl is not None:
        # dry run: no bus, so publish the fixed arm state once
        o = cpp_control.OUT_OFF
        ctl.a[o["state"][0]:o["state"][0] + 6] = state
        ctl.a[o["state_time"][0]] = time.perf_counter()
    rbp.warm_up(ex)
    video = None
    if args.video == "on":
        video = rbp.EpisodeVideo(robot.cameras, Path("/tmp/bench_video.mp4"),
                                 "bench")
        video.start()
    ex.episode()
    if video is not None:
        video.stop()
    if cams:
        for c in cams.values():
            c.disconnect()
    if ctl is not None:
        ctl.stop()
        per = np.diff(ex.cpp_ticks[:, 0]) * 1e3
    else:
        per = np.diff(np.array(ex.ticks)[:, 0]) * 1e3
    nominal = 1000 / args.hz
    late = float(np.mean(per > 1.5 * nominal) * 100)
    lat = np.array(ex.latency) * 1e3
    emit({"bench": "loop", "label": args.label, "model": args.model,
          "trt": args.trt, "compile": args.compile,
          "control": args.control, "video": args.video, "cams": args.cams,
          "tick_ms": stats(per), "late_pct": round(late, 2),
          "plan_ms": stats(lat), "plans": len(lat)}, args.json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bus", "cams", "loop"])
    ap.add_argument("--n", type=int, default=500, help="bus: reads")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--model", default="diffusion_bsp_b")
    ap.add_argument("--trt", default="off")
    ap.add_argument("--compile", default="off")
    ap.add_argument("--control", default="python",
                    choices=["python", "cpp"])
    ap.add_argument("--video", default="off", choices=["off", "on"])
    ap.add_argument("--cams", default="fake", choices=["fake", "real"])
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    {"bus": bench_bus, "cams": bench_cams, "loop": bench_loop}[args.cmd](
        args)


if __name__ == "__main__":
    main()
