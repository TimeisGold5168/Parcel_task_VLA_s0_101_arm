"""Check each recorded episode: video frames and phase marks.

For every episode and camera, the video span in the episode metadata
must hold as many frames as the episode has data rows; a shorter span
means the streaming encoder dropped frames. Also lists the phase marks.

  python outputs/portfolio/parcel_task/check_episodes.py \
    --root ~/.cache/huggingface/lerobot/Low101/parcel-sort-a [--from 7]
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--from", dest="first", type=int, default=0,
                    help="first episode to check")
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
        print(f"ep {ep:3d}: {n / fps:5.1f} s, {marks.get(ep, 0)} marks, "
              f"{status}")
    print(f"{len(eps)} episodes checked, {bad} with missing video frames")


if __name__ == "__main__":
    main()
