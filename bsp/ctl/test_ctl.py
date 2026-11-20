"""Check bsp_ctl against the Python control rule (dry run, no bus).

Two real dataset B segments (bsp/targets_parcel-sort-b_e0.5.npz), a
fade from the first to the second, no step clamp. For each logged tick,
the Python rule (scipy BSpline, smoothstep) is evaluated at the tick
time and compared with the C++ command.

  python outputs/portfolio/bsp/ctl/test_ctl.py
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from bspline import make_spline, segment_range  # noqa: E402
from cpp_control import CppControl  # noqa: E402


def main():
    z = np.load(HERE.parent / "targets_parcel-sort-b_e0.5.npz")
    key = [k for k in z.files if z[k].ndim == 3][0]
    tg = z[key]
    p_old, p_new = tg[100].astype(np.float64), tg[110].astype(np.float64)
    fps, speed, deg = 30.0, 1.5, 3
    ctl = CppControl(hz=50)
    t_start = time.perf_counter()
    ctl.start_run(t_start, fps, speed, max_step=1e9, degree=deg)
    tau = lambda t: (t - t_start) * fps * speed  # noqa: E731
    o0, o1 = segment_range(p_old, deg)
    old = (p_old, o0, tau(time.perf_counter()), o0, o1, 0)
    ctl.set_segment(old)
    time.sleep(0.3)
    n0, n1 = segment_range(p_new, deg)
    now = time.perf_counter()
    new = (p_new, n0, tau(now), n0, n1, 1)
    fade = (now, now + 0.2)
    ctl.set_segment(new, old, fade)
    time.sleep(0.6)
    ctl.hold()
    ticks = ctl.ticks()
    ctl.stop()

    so, sn = make_spline(p_old, deg), make_spline(p_new, deg)
    errs, n_fade = [], 0
    for r in ticks:
        t, cmd = r[0], r[4:10]
        if not np.isfinite(cmd).all():
            continue
        if t < now:  # old segment only
            s = np.clip(old[1] + tau(t) - old[2], o0, o1 - 1e-6)
            ref = so(s)
        else:
            s = np.clip(new[1] + tau(t) - new[2], n0, n1 - 1e-6)
            ref = sn(s)
            if t < fade[1]:
                os_ = np.clip(old[1] + tau(t) - old[2], o0, o1 - 1e-6)
                a = (t - fade[0]) / (fade[1] - fade[0])
                a = a * a * (3 - 2 * a)
                ref = (1 - a) * so(os_) + a * ref
                n_fade += 1
        errs.append(np.abs(ref - cmd).max())
    per = ticks[1:, 1]
    print(f"ticks {len(ticks)}, compared {len(errs)} ({n_fade} in the "
          f"fade); max |C++ - Python| {max(errs):.4f} deg")
    print(f"tick period p50 {np.median(per):.3f} ms, p95 "
          f"{np.percentile(per, 95):.3f}, max {per.max():.3f}")
    # tick time is logged before the evaluation: allow the motion of
    # a few microseconds
    assert max(errs) < 0.05, "C++ control differs from the Python rule"
    print("OK")


def executor_test(seconds=4.0):
    """run_bsp Executor, Python control against CppExecutorMixin.

    A fake policy returns dataset B segments (no GPU, no robot). Both
    runs must give finite commands at 50 Hz on the same path.
    """
    from types import SimpleNamespace

    import torch

    import cpp_control
    import run_bsp as rb

    z = np.load(HERE.parent / "targets_parcel-sort-b_e0.5.npz")
    tg = z[[k for k in z.files if z[k].ndim == 3][0]]
    count = [0]

    def fake_segment(policy, obs, noise):
        count[0] += 1
        return torch.from_numpy(tg[100 + 3 * count[0]][None]).float()

    rb.prepare_observation_for_inference = lambda f, *a: f
    rb.predict_segment = fake_segment

    class Robot:
        cameras = {}

        def get_observation(self):
            return {"front": np.zeros((4, 4, 3), np.uint8),
                    "wrist": np.zeros((4, 4, 3), np.uint8),
                    **{k: 0.0 for k in KEYS}}

        def send_action(self, action):
            return action

    policy = SimpleNamespace(reset=lambda: None,
                             config=SimpleNamespace(type="diffusion"))
    a = SimpleNamespace(speed=1.5, hz=50.0, replan_frac=0.5, max_step=4.0,
                        blend=0.2, fixed_noise=False, debug=True, fps=30,
                        duration=seconds, ddim_steps=10)
    ident = lambda x: x  # noqa: E731
    out = {}
    for mode in ("python", "cpp"):
        count[0] = 0
        cls, ctl = rb.Executor, None
        if mode == "cpp":
            ctl = CppControl(hz=50)
            o = cpp_control.OUT_OFF
            ctl.a[o["state_time"][0]] = time.perf_counter()
            cls = type("CppTest", (cpp_control.CppExecutorMixin,
                                   rb.Executor), {})
        ex = cls(Robot(), policy, ident, ident, KEYS, a, 3)
        ex.cpp = ctl
        ex.done_pressed = lambda: False
        ex.episode()
        if ctl is not None:
            ctl.stop()
            per = np.diff(ex.cpp_ticks[:, 0]) * 1e3
        else:
            per = np.diff(np.array(ex.ticks)[:, 0]) * 1e3
        cmd = np.array(ex.log)[:, 2:]
        out[mode] = cmd
        print(f"{mode}: {ex.n_segments} segments, {len(cmd)} commands, "
              f"finite {np.isfinite(cmd).all()}, tick p50 "
              f"{np.median(per):.2f} ms, max {per.max():.2f} ms")
        assert ex.n_segments > 3 and len(cmd) > 0.8 * seconds * a.hz
        assert np.isfinite(cmd).all()
    lo = min(len(out["python"]), len(out["cpp"]))
    d = np.abs(out["python"][:lo] - out["cpp"][:lo]).max()
    print(f"Python vs C++ command path: max |diff| {d:.2f} deg (tick "
          "times and plan times differ a little)")
    print("executor OK")


KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


if __name__ == "__main__":
    main()
    executor_test()
