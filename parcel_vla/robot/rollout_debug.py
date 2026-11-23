"""Compare rollout debug logs (run_parcel.sh DEBUG=1), one column each.

  python -m parcel_vla.robot.rollout_debug \
    runs/debug_logs/*.npz

Arm joints 0-4, max over joints, degrees per 30 Hz tick.
- pause: ticks longer than 2 periods (sync: the robot waits for a plan).
- step: command change per tick (the motion).
- accel: 2nd difference of the command (cmd) or of the measured joint
  position (meas); "switch" = the 3 ticks after a new chunk starts.
- RTC only: guidance error = new chunk (aligned by the actions really
  used during the plan) minus the next action of the old chunk; skip =
  extra jump from real_delay > actions used.
"""

import sys
from pathlib import Path

import numpy as np

ARM = slice(0, 5)
PERIOD = 1 / 30


def p(x, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.percentile(x, q) if len(x) else np.nan


def accel(t, q, switch_t):
    """Max-joint |2nd diff| per tick; mask for ticks near a switch."""
    a = np.abs(np.diff(q[:, ARM], 2, axis=0)).max(1)
    tc = t[1:-1]
    near = np.zeros(len(tc), bool)
    for s in switch_t:
        i = np.searchsorted(tc, s)
        near[i:i + 3] = True
    return a, near


def analyse(path):
    d = np.load(path, allow_pickle=True)
    r = {"label": str(d["label"]) or Path(path).stem}
    st, sq = d["send_t"], d["send_q"]
    plan = d["plan"]
    merges = d["merge"]
    rtc = len(merges) > 0
    switch_t = [m[0] for m in merges] if rtc else list(plan[:, 1])
    dt = np.diff(st)
    r["mode"] = "rtc" if rtc else "sync"
    r["run time (s)"] = st[-1] - st[0]
    r["tick p50 (ms)"] = 1000 * np.median(dt)
    r["tick p95 (ms)"] = 1000 * p(dt, 95)
    pauses = dt[dt > 2 * PERIOD]
    r["pauses"] = len(pauses)
    r["pause time (%)"] = 100 * pauses.sum() / r["run time (s)"]
    lat = 1000 * (plan[:, 1] - plan[:, 0])
    r["plans"] = len(plan)
    r["plan p50 (ms)"] = np.median(lat[1:])
    r["plan p95 (ms)"] = p(lat[1:], 95)
    r["plan max (ms)"] = lat[1:].max()
    r["chunk interval (s)"] = np.median(np.diff(switch_t))
    step = np.abs(np.diff(sq[:, ARM], axis=0)).max(1)
    r["cmd step p50"] = np.median(step)
    r["cmd step p95"] = p(step, 95)
    a, near = accel(st, sq, switch_t)
    r["cmd accel inside p95"] = p(a[~near], 95)
    r["cmd accel switch p95"] = p(a[near], 95)
    r["cmd switch/inside"] = r["cmd accel switch p95"] / \
        r["cmd accel inside p95"]
    ot, oq = d["obs_t"], d["obs_q"]
    if len(ot) > 3:
        a, near = accel(ot, oq, switch_t)
        r["meas accel inside p95"] = p(a[~near], 95)
        r["meas accel switch p95"] = p(a[near], 95)
    if rtc:
        guide, skip, real, used = [], [], [], []
        for _, rd, u, old, new in merges:
            if old is None or u < 0 or not len(old):
                continue
            real.append(rd)
            used.append(u)
            u, rd = min(u, len(new) - 1), min(rd, len(new) - 1)
            guide.append(np.abs(new[u, ARM] - old[0, ARM]).max())
            skip.append(np.abs(new[rd, ARM] - new[u, ARM]).max())
        r["rtc real_delay p50"] = np.median(real)
        r["rtc used p50"] = np.median(used)
        r["rtc guidance err p50"] = np.median(guide)
        r["rtc guidance err p95"] = p(guide, 95)
        r["rtc skip p95"] = p(skip, 95)
    return r


def main(paths):
    runs = [analyse(x) for x in paths]
    names = []
    for r in runs:
        names += [k for k in r if k not in names]
    w = max(12, *(len(str(r["label"])) for r in runs))
    print(f"{'':24s}" + "".join(f"{str(r['label'])[:w]:>{w + 2}s}"
                                for r in runs))
    for k in names[1:]:
        row = f"{k:24s}"
        for r in runs:
            v = r.get(k, "")
            row += f"{v:>{w + 2}.2f}" if isinstance(v, (float, np.floating)) \
                else f"{str(v):>{w + 2}s}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
