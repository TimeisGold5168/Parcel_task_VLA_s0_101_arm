"""Python side of the C++ control process bsp_ctl (item 11).

bsp_ctl owns the servo bus and runs the B-spline loop at a fixed rate.
Python plans and writes each new segment into shared memory
(/dev/shm/bsp_ctl), and reads the arm state and the last command back.
The GIL of the Python process then cannot delay a control tick.

  ctl = CppControl(port=PORT, calib=CALIB)   # port=None: dry run
  ctl.start_run(t_start, fps, speed, max_step, degree)
  ctl.set_segment(params, start, now0, t0, t1, seg_id, old, fade)
  ctl.last_cmd(); ctl.state(); ctl.ticks(); ctl.stop()

CppExecutorMixin puts this under a run_bsp Executor (--control cpp in
run_bsp_parcel.py). Build: make -C parcel_vla/inference_improvement/ctl
"""

import mmap
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BIN = HERE / "ctl" / "bsp_ctl"
SHM = Path("/dev/shm/bsp_ctl")
CALIB = (Path.home() / ".cache/huggingface/lerobot/calibration/robots/"
         "so_follower/my_follower_arm.json")

# layout.h, all float64
MAX_ROWS, MAX_DIM, LOG_N = 32, 8, 8192
LOG_COLS = 4 + MAX_DIM  # tick start, period, bus write, bus read, cmd
SEG = ["seg_id", "rows", "dim", "start", "now0", "t0", "t1",
       ("knots", MAX_ROWS), ("ctrl", MAX_ROWS * MAX_DIM)]
IN = ["seq", "mode", "t_start", "fps", "speed", "max_step", "degree",
      "fade_on", "fade_t0", "fade_t1", ("direct", MAX_DIM),
      ("seg", SEG), ("old", SEG)]
OUT = ["seq", "alive", "pid", "n_ticks", "read_errors", "state_time",
       "cmd_time", "cmd_seg", ("state", MAX_DIM), ("last_cmd", MAX_DIM)]
HOLD, RUN, DIRECT, QUIT = 0, 1, 2, 3


def _offsets(fields, base=0):
    off, pos = {}, base
    for f in fields:
        if isinstance(f, str):
            off[f] = (pos, 1)
            pos += 1
        elif isinstance(f[1], list):
            sub, end = _offsets(f[1], pos)
            off[f[0]] = (pos, end - pos, sub)
            pos = end
        else:
            off[f[0]] = (pos, f[1])
            pos += f[1]
    return off, pos


IN_OFF, IN_END = _offsets(IN)
OUT_OFF, OUT_END = _offsets(OUT, IN_END)
LOG_OFF = OUT_END
TOTAL = LOG_OFF + LOG_N * LOG_COLS


class CppControl:
    def __init__(self, hz=50.0, port=None, calib=CALIB, read_every=1,
                 rt=False, shm=SHM):
        if not BIN.exists():
            raise SystemExit(f"{BIN} missing: make -C {BIN.parent}")
        self.shm_path = Path(shm)
        with open(self.shm_path, "wb") as f:
            f.truncate(TOTAL * 8)
        fd = os.open(self.shm_path, os.O_RDWR)
        self.mm = mmap.mmap(fd, TOTAL * 8)
        os.close(fd)
        self.a = np.frombuffer(self.mm, dtype=np.float64)
        self.dim = 6
        cmd = [str(BIN), "--shm", str(self.shm_path), "--hz", str(hz),
               "--read-every", str(read_every)]
        if port:
            cmd += ["--port", str(port), "--calib", str(calib)]
        if rt:
            cmd.append("--rt")
        self.proc = subprocess.Popen(cmd)
        t = time.perf_counter()
        while self._out("alive") != 1:
            if self.proc.poll() is not None or time.perf_counter() - t > 3:
                raise RuntimeError("bsp_ctl did not start")
            time.sleep(0.01)

    # -- raw access
    def _in(self, name, value=None, off=IN_OFF):
        o = off[name]
        if value is None:
            return self.a[o[0]] if o[1] == 1 else self.a[o[0]:o[0] + o[1]]
        if o[1] == 1:
            self.a[o[0]] = value
        else:
            v = np.asarray(value, np.float64).ravel()
            self.a[o[0]:o[0] + len(v)] = v

    def _out(self, name):
        o = OUT_OFF[name]
        return self.a[o[0]] if o[1] == 1 else self.a[o[0]:o[0] + o[1]]

    def _write(self, fn):
        """Seqlock writer: seq odd while the block changes."""
        s = IN_OFF["seq"][0]
        self.a[s] += 1
        fn()
        self.a[s] += 1

    def _read_out(self, names):
        s = OUT_OFF["seq"][0]
        for _ in range(100):
            a = self.a[s]
            if int(a) % 2:
                continue
            vals = [np.array(self._out(n)) for n in names]
            if self.a[s] == a:
                return vals
        return None

    # -- control
    def start_run(self, t_start, fps, speed, max_step, degree):
        def fn():
            self._in("t_start", t_start)
            self._in("fps", fps)
            self._in("speed", speed)
            self._in("max_step", max_step)
            self._in("degree", degree)
            self._in("fade_on", 0)
            self.a[IN_OFF["seg"][2]["seg_id"][0]] = -1
            self._in("mode", RUN)
        self._write(fn)

    def hold(self):
        self._write(lambda: self._in("mode", HOLD))

    def direct(self, target):
        def fn():
            self._in("direct", target)
            self._in("mode", DIRECT)
        self._write(fn)

    def _put_seg(self, key, params, start, now0, t0, t1, seg_id, degree):
        sub = IN_OFF[key][2]
        p = np.asarray(params, np.float64)
        rows, dim = len(p), p.shape[1] - 1
        knots = np.maximum.accumulate(p[:, 0])
        ctrl = np.zeros((MAX_ROWS, MAX_DIM))
        ctrl[:rows, :dim] = p[:, 1:]
        for k, v in (("seg_id", seg_id), ("rows", rows), ("dim", dim),
                     ("start", start), ("now0", now0), ("t0", t0),
                     ("t1", t1)):
            self.a[sub[k][0]] = v
        self.a[sub["knots"][0]:sub["knots"][0] + rows] = knots
        self.a[sub["ctrl"][0]:sub["ctrl"][0] + MAX_ROWS * MAX_DIM] = \
            ctrl.ravel()

    def set_segment(self, seg, old=None, fade=None):
        """seg / old: (params, start, now0, t0, t1, seg_id)."""
        def fn():
            self._put_seg("seg", *seg[:6], None)
            if old is not None and fade is not None:
                self._put_seg("old", *old[:6], None)
                self._in("fade_t0", fade[0])
                self._in("fade_t1", fade[1])
                self._in("fade_on", 1)
            else:
                self._in("fade_on", 0)
        self._write(fn)

    def last_cmd(self):
        r = self._read_out(["last_cmd", "cmd_time"])
        if r is None or r[1] == 0:
            return None
        return r[0][:self.dim].copy()

    def state(self):
        r = self._read_out(["state", "state_time"])
        if r is None or r[1] == 0:
            return None, 0.0
        return r[0][:self.dim].copy(), float(r[1])

    def ticks(self):
        """(n, 12): tick start (s), period, bus write, bus read (ms),
        command (deg, NaN when no command)."""
        n = int(self._out("n_ticks"))
        log = self.a[LOG_OFF:LOG_OFF + LOG_N * LOG_COLS].reshape(
            LOG_N, LOG_COLS)
        if n <= LOG_N:
            return log[:n].copy()
        i = n % LOG_N
        return np.concatenate([log[i:], log[:i]])

    def stop(self):
        self._write(lambda: self._in("mode", QUIT))
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class CppExecutorMixin:
    """Executor with the control loop in bsp_ctl (put it first in the
    bases). The planner thread is the same; the control thread only
    copies the command log back from the shared memory.
    """

    cpp = None

    def plan(self, frame, tau_obs, state):
        from parcel_vla.robot import run_bsp as rb

        if self.degree is None:
            raise SystemExit("--control cpp needs a B-spline policy")
        self.last_cmd = self.cpp.last_cmd()
        got = {}
        make_curve = rb.make_curve

        def keep(out, degree):
            got["p"] = out
            return make_curve(out, degree)

        rb.make_curve = keep
        try:
            super().plan(frame, tau_obs, state)
        finally:
            rb.make_curve = make_curve
        with self.seg_lock:
            seg, fade = self.seg, self.fade
        if not hasattr(self, "seg_params"):
            self.seg_params = {}
        self.seg_params[seg[5]] = got["p"]
        old = fade_t = None
        if fade is not None:
            o = fade[0]
            old = (self.seg_params[o[5]], *o[1:5], o[5])
            fade_t = (fade[1], fade[2])
        self.cpp.set_segment((got["p"], *seg[1:5], seg[5]), old, fade_t)
        for k in [k for k in self.seg_params if k < seg[5] - 2]:
            del self.seg_params[k]

    def control(self):
        a = self.args
        self.seg_params = {}
        n0 = int(self.cpp._out("n_ticks"))
        self.cpp.start_run(self.t_start, self.fps, a.speed, a.max_step,
                           self.degree)
        while self.running:
            time.sleep(0.05)
        self.cpp.hold()
        self.last_cmd = self.cpp.last_cmd()
        n = int(self.cpp._out("n_ticks")) - n0
        t = self.cpp.ticks()[-n:] if n > 0 else np.zeros((0, LOG_COLS))
        self.cpp_ticks = t
        fps_s = self.fps * a.speed
        self.log = [[r[0] - self.t_start, (r[0] - self.t_start) * fps_s,
                     *r[4:4 + len(self.keys)]]
                    for r in t if np.isfinite(r[4])]

    def observe(self):
        state, _ = self.cpp.state()
        obs = {k: float(v) for k, v in zip(self.keys, state)}
        for c, cam in self.robot.cameras.items():
            obs[c] = cam.read_latest()
        frame = {"observation.state": np.asarray(state, np.float32)}
        for c in self.robot.cameras:
            frame[f"observation.images.{c}"] = obs[c]
        obs["front"] = obs.get("top", obs.get("front"))
        return obs, np.asarray(state, np.float32), frame


def go_to_cpp(ex, pose, seconds=2.5, hz=50):
    """run_bsp.go_to through bsp_ctl (direct mode, then hold)."""
    start, _ = ex.cpp.state()
    n = int(seconds * hz)
    for i in range(1, n + 1):
        ex.cpp.direct(start + (np.asarray(pose) - start) * i / n)
        time.sleep(1 / hz)
    time.sleep(0.1)
    ex.cpp.hold()
