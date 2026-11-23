"""Check each recorded episode: video frames and phase marks.

For every episode and camera, the video span in the episode metadata
must hold as many frames as the episode has data rows; a shorter span
means the streaming encoder dropped frames. Also lists the phase marks.
--home a1 checks that each episode starts and ends at the A1 home pose;
--home median checks them against the median start pose of the
dataset (one ready pose for all episodes). Tolerance: --home-tol deg.

  python -m parcel_vla.data.check_episodes \
    --root ~/.cache/huggingface/lerobot/Low101/parcel-sort-a [--from 7]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# A1 median start / end pose: pan, lift, elbow, wrist flex, wrist roll
HOME = np.array([-6.0, -104.0, 90.5, 52.4, -1.8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--from", dest="first", type=int, default=0,
                    help="first episode to check")
    ap.add_argument("--home", choices=["a1", "median"],
                    help="check start and end pose: A1 home or the "
                    "median start pose of this dataset")
    ap.add_argument("--home-tol", type=float, default=20.0,
                    help="deg; A1 p95 is about 20")
    args = ap.parse_args()

    root = args.root.expanduser()
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in files])
    eps = eps[eps.episode_index >= args.first]
    cams = sorted({c.split("/")[1] for c in eps.columns
                   if c.startswith("videos/")})
    marks = {}
    if (root / "phase_marks.csv").exists():
        m = pd.read_csv(root / "phase_marks.csv")
        marks = m.groupby("episode").size().to_dict()

    poses = {}
    if args.home:
        data = pd.concat(pd.read_parquet(f) for f in
                         sorted((root / "data").rglob("*.parquet")))
        ends = {}
        for ep, g in data.groupby("episode_index"):
            s = np.stack(g["observation.state"].values)[:, :5]
            ends[int(ep)] = (s[0], s[-1])
        ref = HOME
        if args.home == "median":
            ref = np.median([a for a, _ in ends.values()], axis=0)
            print("ready pose (median start): " + " ".join(
                f"{v:.0f}" for v in ref))
        for ep, (a, b) in ends.items():
            if ep >= args.first:
                poses[ep] = (np.abs(a - ref).max(), np.abs(b - ref).max())

    bad = 0
    for _, r in eps.iterrows():
        ep, n = int(r.episode_index), int(r.length)
        short = []
        for c in cams:
            span = (r[f"videos/{c}/to_timestamp"]
                    - r[f"videos/{c}/from_timestamp"])
            missing = n - round(span * fps)
            if missing > 0:
                short.append(f"{c.split('.')[-1]} -{missing}")
        bad += bool(short)
        status = "MISSING FRAMES " + ", ".join(short) if short else "ok"
        if ep in poses:
            d0, d1 = poses[ep]
            off = [f"{w} {d:.0f} deg" for w, d in (("start", d0),
                                                   ("end", d1))
                   if d > args.home_tol]
            if off:
                status += "; OFF START POSE: " + ", ".join(off)
                bad += not short
        print(f"ep {ep:3d}: {n / fps:5.1f} s, {marks.get(ep, 0)} marks, "
              f"{status}")
    print(f"{len(eps)} episodes checked, {bad} with problems")


if __name__ == "__main__":
    main()
