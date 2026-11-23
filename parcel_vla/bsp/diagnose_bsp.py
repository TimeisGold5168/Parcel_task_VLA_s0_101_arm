"""Diagnose BSP executor runs from the ep<n>_debug.npz logs.

Reads the per-tick and per-plan logs that run_bsp.py --debug saves and
prints one row per episode, plus tuning hints.

Usage:
  python -m parcel_vla.bsp.diagnose_bsp \
    runs/bsp/sweep_diffusion_SQ_<stamp>

Columns:
  bl, fn  --blend (s) and --fixed-noise (y / n)
  plan    mean / p95 planning time (ms)
  margin  min time left on the old segment when the new one arrives (s);
          below 0 the arm waited at the end of a segment
  hold    % of ticks frozen at a segment end (plan too late)
  clamp   % of ticks where --max-step cut a joint step (worst joint)
  seamC   share of those clamps within 3 ticks of a segment switch
  seam    p95 jump between the old command and the new segment, arm
          joints (deg; runs before 2026-10-25 21:00 include the gripper)
  shift   median |start - natural| on the new segment (demo frames)
  late    % of control ticks later than 1.5 periods
  lag     max |measured - commanded| at plan time, arm joints (deg;
          older runs include the gripper, which lags while it grips)
  vel     p99 commanded speed, worst joint (deg/s)
  rev/s   arm direction reversals per second (jerk indicator)
"""

import sys
from pathlib import Path

import numpy as np

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
ARM = slice(0, 5)


def summarize(path):
    d = np.load(path)
    ticks, plans = d["ticks"], d["plans"]
    hz, fps, speed = float(d["hz"]), float(d["fps"]), float(d["speed"])
    max_step = float(d["max_step"])
    out = {"speed": speed, "max_step": max_step,
           "replan_frac": float(d["replan_frac"]),
           "ddim": int(d["ddim_steps"]),
           "blend": float(d["blend"]) if "blend" in d else 0.0,
           "fixed_noise": bool(d["fixed_noise"]) if "fixed_noise" in d
           else False,
           "n_ticks": len(ticks),
           "n_plans": len(plans)}
    if len(ticks) < 10 or len(plans) < 2:
        return out
    t, sid, s_raw, t1 = ticks[:, 0], ticks[:, 2], ticks[:, 3], ticks[:, 4]
    wait, raw, cmd = ticks[:, 5], ticks[:, 6:12], ticks[:, 12:18]

    lat = plans[:, 2] * 1000
    margin_s = plans[1:, 7] / (fps * speed)
    out.update(plan_ms=float(lat.mean()),
               plan_p95=float(np.percentile(lat, 95)),
               margin_min=float(np.nanmin(margin_s)),
               late_plans=int((margin_s < 0).sum()))

    out["hold_pct"] = float(100 * (s_raw >= t1 - 1e-3).mean())

    clamped = np.abs(raw[1:] - cmd[:-1]) > max_step - 1e-6
    per_joint = 100 * clamped.mean(0)
    worst = int(per_joint.argmax())
    switch = np.flatnonzero(np.diff(sid) != 0)
    near = np.zeros(len(clamped), bool)
    for k in switch:
        near[k:k + 3] = True
    any_c = clamped.any(1)
    out.update(clamp_pct=float(per_joint.max()), clamp_joint=JOINTS[worst],
               seam_clamp=float(100 * (any_c & near).sum()
                                / max(any_c.sum(), 1)))

    seam = plans[1:, 8]
    shift = np.abs(plans[1:, 4] - plans[1:, 3])
    out.update(seam_p95=float(np.nanpercentile(seam, 95)),
               shift_med=float(np.nanmedian(shift)),
               lag_max=float(np.nanmax(plans[1:, 9])))

    dt = np.diff(t)
    out.update(late_pct=float(100 * (dt > 1.5 / hz).mean()),
               dt_p99_ms=float(1000 * np.percentile(dt, 99)),
               wait_p99_ms=float(1000 * np.percentile(wait, 99)))

    vel = np.diff(cmd, axis=0) / dt[:, None].clip(1e-3)
    v99 = np.percentile(np.abs(vel), 99, axis=0)
    out.update(vel_p99=float(v99.max()), vel_joint=JOINTS[int(v99.argmax())])
    v = vel[:, ARM]
    moving = np.abs(v) > 5.0
    flips = (np.sign(v[1:]) != np.sign(v[:-1])) & moving[1:] & moving[:-1]
    out["rev_per_s"] = float(flips.sum() / max(t[-1] - t[0], 1e-3))
    return out


def tuning_hints(rows):
    msgs = []
    if any(r.get("clamp_pct", 0) > 3 for r in rows):
        seamy = np.mean([r.get("seam_clamp", 0) for r in rows])
        if seamy > 50:
            msgs.append("clamps happen mostly at segment switches: the new "
                        "segment starts away from the arm. Lower "
                        "--replan-frac (more overlap, e.g. 0.3).")
        else:
            msgs.append("clamps happen inside segments: the curve is "
                        "faster than --max-step allows. Raise --max-step "
                        "(5-6 deg at 50 Hz = 250-300 deg/s).")
    if any(r.get("hold_pct", 0) > 1 or r.get("late_plans", 0) > 0
           for r in rows):
        msgs.append("plans arrive after the segment ends (hold > 1% or "
                    "margin < 0): lower --replan-frac, or faster plans "
                    "with --ddim-steps 5.")
    if any(r.get("late_pct", 0) > 5 for r in rows):
        msgs.append("the control loop misses ticks while planning (late "
                    "> 5%): CPU/GIL contention. Try --hz 30.")
    if any(r.get("lag_max", 0) > 15 for r in rows):
        msgs.append("the arm lags the command by more than 15 deg: the "
                    "servos are at their speed limit. More speed will not "
                    "help on those moves.")
    if any(r.get("vel_p99", 0) > 250 for r in rows):
        msgs.append("commanded speed above 250 deg/s: near the servo top "
                    "speed.")
    return msgs or ["no problem found by the thresholds."]


def print_table(rows, hints=False):
    head = (f"{'speed':>5} {'step':>4} {'bl':>4} {'fn':>2} {'plan':>9} "
            f"{'margin':>6} "
            f"{'hold':>5} {'clamp':>11} {'seamC':>5} {'seam':>5} "
            f"{'shift':>5} {'late':>5} {'lag':>5} {'vel':>10} {'rev/s':>5}"
            f" {'ok':>2} {'done':>5}")
    print(head)
    for r in rows:
        if "plan_ms" not in r:
            print(f"{r['speed']:>5g}  (too short to analyse)")
            continue
        print(f"{r['speed']:>5g} {r['max_step']:>4g} {r['blend']:>4g} "
              f"{'y' if r['fixed_noise'] else 'n':>2} "
              f"{r['plan_ms']:>4.0f}/{r['plan_p95']:<4.0f} "
              f"{r['margin_min']:>6.2f} {r['hold_pct']:>5.1f} "
              f"{r['clamp_pct']:>5.1f} {r['clamp_joint']:<5} "
              f"{r['seam_clamp']:>5.0f} {r['seam_p95']:>5.1f} "
              f"{r['shift_med']:>5.1f} {r['late_pct']:>5.1f} "
              f"{r['lag_max']:>5.1f} {r['vel_p99']:>4.0f} "
              f"{r['vel_joint']:<5} {r['rev_per_s']:>5.1f} "
              f"{str(r.get('success', '')):>2} {str(r.get('done_s', '')):>5}")
    if hints:
        print("\nhints:")
        for m in tuning_hints(rows):
            print(" -", m)


def main():
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        files += sorted(p.rglob("*_debug.npz")) if p.is_dir() else [p]
    rows = []
    for f in files:
        r = summarize(f)
        r["file"] = str(f)
        rows.append(r)
    rows.sort(key=lambda r: r["speed"])
    print_table(rows, hints=True)


if __name__ == "__main__":
    main()
