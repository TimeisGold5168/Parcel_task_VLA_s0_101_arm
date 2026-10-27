"""Build one frame grid per episode for a vision-model judge.

Each tile = front image (top) + wrist image (bottom), sampled every
--every seconds of wall-clock time, plus the gripper key moments from
the joint signal (first close, first hold, last release), which a
fixed interval can miss. Labels show time and the key moment only, not
the auto-score, so the judge is not biased.

Usage:
  python outputs/portfolio/vlm_frames.py rollout_diffusion_SQ [--every 2]
Output: spot_review/vlm/<dataset>/ep<n>.jpg  (n = 0-based episode)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import review_rollouts as R  # noqa: E402

COLS = 6
TILE_W, TILE_H = 320, 240


def tile(ds, i, label, key):
    front = cv2.resize(R.rgb(ds, i), (TILE_W, TILE_H))
    wrist = cv2.resize(R.rgb(ds, i, "observation.images.wrist"),
                       (TILE_W, TILE_H))
    t = cv2.cvtColor(np.vstack([front, wrist]), cv2.COLOR_RGB2BGR)
    color = (0, 140, 255) if key else (0, 0, 0)
    cv2.rectangle(t, (0, 0), (TILE_W, 24), color, -1)
    cv2.putText(t, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    return t


def frames(n, hz, every, keys):
    """Sorted (frame, label, is_key) list."""
    picks = {}
    t = 0.0
    while round(t * hz) < n:
        picks[int(round(t * hz))] = (f"t={t:.0f}s", False)
        t += every
    picks[n - 1] = (f"t={(n - 1) / hz:.1f}s end", False)
    for name, k in keys:
        if k is not None:
            picks[k] = (f"t={k / hz:.1f}s {name}", True)
    return [(k, *picks[k]) for k in sorted(picks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--every", type=float, default=2.0)
    args = ap.parse_args()
    name = R.newest(args.prefix)
    ds, s, a, e = R.load(name)
    out = R.OUT / "vlm" / name
    out.mkdir(parents=True, exist_ok=True)
    for ep in np.unique(e):
        sel = np.flatnonzero(e == ep)
        hz = min(len(sel) / R.EPISODE_S, float(R.FPS))
        close, hold, rel = R.gripper_events(s[sel], a[sel])["_idx"]
        keys = [("gripper closes", close), ("gripper holds", hold),
                ("gripper opens", rel)]
        tiles = [tile(ds, sel[k], lab, key)
                 for k, lab, key in frames(len(sel), hz, args.every, keys)]
        while len(tiles) % COLS:
            tiles.append(np.full_like(tiles[0], 40))
        rows = [np.hstack(tiles[i:i + COLS])
                for i in range(0, len(tiles), COLS)]
        path = out / f"ep{ep}.jpg"
        cv2.imwrite(str(path), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{path}  {len(tiles)} tiles, {hz:.1f} Hz")


if __name__ == "__main__":
    main()
