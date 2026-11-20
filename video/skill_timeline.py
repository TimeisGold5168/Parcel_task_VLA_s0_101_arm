"""Rebuild the live skill text of a rollout episode from the logs.

The live preview showed "skill: <name>", but the dataset videos do not
have it. The planner log has the wall time, the text in use and the
gripper at each call. The skill log has the launch skill and the key
presses. This script finds the episode start on the wall clock: the
shift that makes the logged gripper match the recorded gripper. Then
it gives the skill changes in episode time.

Run: python outputs/portfolio/video/skill_timeline.py
Output: outputs/portfolio/video/footage/skill_timeline.csv
"""

import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
LOGS = Path(__file__).resolve().parents[1] / "parcel_task/debug_logs"
OUT = Path(__file__).resolve().parent / "footage/skill_timeline.csv"
FPS = 30
DRIVE = "rollout_bench_parcel_smolvla_b_sk3d_planner_drive"
# planner/skill log stem: episodes of the drive dataset in that session
SESSIONS = {"20261119_201229": [0], "20261119_202451": [1],
            "20261119_202623": [2], "20261119_204507": [3, 4, 5],
            "20261119_205901": [6], "20261119_210259": [7, 8]}


def gripper(ds):
    files = sorted(glob.glob(str(LOW / ds / "data/chunk-000/*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files])
    return {ep: np.stack(g["observation.state"].to_numpy())[:, 5]
            for ep, g in d.groupby("episode_index")}


def read_log(stem):
    rows = list(csv.DictReader(open(LOGS / f"planner_{stem}.csv")))
    keys = [line.strip().split(",") for line in
            open(LOGS / f"skill_{stem}.csv") if line.strip()]
    return rows, [(float(t), k) for t, k in keys]


def align(g, rows, after):
    """Wall time of frame 0 with the best gripper match after `after`."""
    t = np.array([float(r["time"]) for r in rows])
    v = np.array([float(r["gripper"]) for r in rows])
    dur = len(g) / FPS
    best = (np.inf, None, 0)
    for s in np.arange(after, t[-1], 0.1):
        m = (t >= s) & (t < s + dur)
        if m.sum() < 5:
            continue
        idx = ((t[m] - s) * FPS).astype(int)
        err = np.abs(g[idx] - v[m]).mean()
        if err < best[0]:
            best = (err, s, int(m.sum()))
    return best


def changes(rows, keys):
    """(wall time, skill, source) for each change of the live skill."""
    ev = [(t, k.split()[-1], "at start" if k.startswith("launch") else
           "human input") for t, k in keys
          if k.split()[-1] in ("bag", "box", "top")]
    for a, b in zip(rows, rows[1:]):
        if a["call"] == "plan" and b["text_in_use"] == a["skill"] \
                and a["skill"] != a["text_in_use"]:
            ev.append((float(a["time"]), a["skill"], "planner"))
        elif a["call"] == "verify" and b["text_in_use"] == a["skill"] \
                and a["skill"] != a["text_in_use"]:
            ev.append((float(a["time"]), a["skill"], "planner, wrist"))
    ev.sort()
    out, now = [], None
    for t, s, src in ev:
        if s != now:
            out.append((t, s, src))
            now = s
    return out


def main():
    grip = gripper(DRIVE)
    lines = [("episode", "t_s", "skill", "source", "match_err", "samples")]
    for stem, eps in SESSIONS.items():
        rows, keys = read_log(stem)
        ev = changes(rows, keys)
        after = keys[0][0]
        for ep in eps:
            g = grip[ep]
            err, s, n = align(g, rows, after)
            if s is None:
                print(f"ep{ep}: no planner calls in range")
                continue
            after = s + len(g) / FPS
            now = [e for e in ev if e[0] <= s]
            start = now[-1][1] if now else ""
            print(f"ep{ep}: start {s:.1f}, gripper error {err:.2f} "
                  f"({n} samples), skill at start: {start}")
            lines.append((ep, "0.0", start, "at start", f"{err:.2f}", n))
            for t, sk, src in ev:
                if s < t < s + len(g) / FPS:
                    lines.append((ep, f"{t - s:.1f}", sk, src, "", ""))
                    print(f"   {t - s:6.1f} s  {sk:4s} ({src})")
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerows(lines)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
