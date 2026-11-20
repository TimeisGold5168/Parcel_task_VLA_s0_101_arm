"""Frame sheets and a score template for a benchmark dataset.

    python bench_sheets.py rollout_bench_parcel_smolvla_b_all_heap3

Writes to outputs/portfolio/bench/<dataset>/:
  e<N>.jpg     16 top-camera frames, evenly spaced over the episode
  starts.jpg   the tray (top and side camera) at the start of each
               episode, to read the parcel kinds of each heap
  scores.csv   one row per episode: length, gripper cycles and time
               still at home (home_s) from the data; the outcome
               columns are empty (fill them in after looking at the
               sheets and the video). cases: edge cases seen, e.g.
               "box_on_box far_edge on_table"
"""

import argparse
import glob
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
OUT = Path(__file__).resolve().parent.parent / "bench"
COLS = ["episode", "time_s", "grip_cycles", "home_s", "parcels",
        "cleared", "wrong_label", "help", "cases", "note"]
# tray area at the start: (rows, cols) of the top and side images
TRAY = {"top": (slice(0, 170), slice(80, 420)),
        "side": (slice(0, 330), slice(250, 640))}


def grip_cycles(grip, thr=20.0):
    """Count gripper openings (state above thr after being below)."""
    opened = grip > thr
    return int(np.sum(opened[1:] & ~opened[:-1]))


def home_time(state, fps=30):
    """Seconds the arm is still at the rest pose (shoulder lift below
    -95 deg, elbow above 85 deg, joint speed below 10 deg/s)."""
    speed = np.abs(np.diff(state[:, :5], axis=0)).sum(1) * fps
    speed = np.convolve(np.r_[0, speed], np.ones(15) / 15, mode="same")
    still = (state[:, 1] < -95) & (state[:, 2] > 85) & (speed < 10)
    return round(float(still.sum()) / fps, 1)


def frames_at(root, row, cam, ts):
    """Frames (BGR) of one camera at the episode times ts (s)."""
    k = f"videos/observation.images.{cam}/"
    f = (root / k / f"chunk-{int(row[k + 'chunk_index']):03d}"
         / f"file-{int(row[k + 'file_index']):03d}.mp4")
    t0 = row[k + "from_timestamp"]
    want = [t0 + t for t in ts]
    c = av.open(str(f))
    c.seek(int(max(want[0] - 1.0, 0) * 1e6))
    out, i = [], 0
    for x in c.decode(video=0):
        t = float(x.pts * x.time_base)
        while i < len(want) and t >= want[i] - 1e-3:
            out.append(x.to_ndarray(format="bgr24"))
            i += 1
        if i >= len(want):
            break
    c.close()
    return out


def sheet(root, row, path, n=16, cam="top"):
    dur = row["length"] / 30.0
    ts = np.linspace(0.1, dur - 0.1, n)
    frames = []
    for t, im in zip(ts, frames_at(root, row, cam, ts)):
        im = cv2.resize(im, (320, 180))
        cv2.putText(im, f"e{int(row.episode_index)} {t:.0f}s",
                    (5, 20), 0, 0.6, (0, 255, 255), 2)
        frames.append(im)
    while len(frames) < n:
        frames.append(np.zeros((180, 320, 3), np.uint8))
    rows = [np.hstack(frames[j:j + 4]) for j in range(0, n, 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def start_tile(root, row):
    parts = []
    for cam, crop in TRAY.items():
        im = frames_at(root, row, cam, [0.2])[0][crop]
        parts.append(cv2.resize(im, (340, 200)))
    tile = np.hstack(parts)
    cv2.putText(tile, f"e{int(row.episode_index)}", (5, 25), 0, 0.8,
                (0, 255, 255), 2)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--parcels", type=int, default=0,
                    help="parcels in each heap (prefills the score)")
    a = ap.parse_args()
    root = LOW / a.dataset
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    data = pd.concat(pd.read_parquet(f) for f in
                     sorted(glob.glob(f"{root}/data/*/*.parquet")))
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")))
    old = out / "scores.csv"
    done = (pd.read_csv(old, dtype=str, keep_default_na=False)
            if old.exists() else pd.DataFrame(columns=COLS))
    rows, tiles = [], []
    for _, r in eps.sort_values("episode_index").iterrows():
        e = int(r.episode_index)
        g = data[data.episode_index == e]
        state = np.stack(g["observation.state"].values)
        if not (out / f"e{e}.jpg").exists():
            sheet(root, r, out / f"e{e}.jpg")
        tiles.append(start_tile(root, r))
        prev = done[done.episode.astype(int) == e]
        if len(prev):
            rows.append({**prev.iloc[0].to_dict(),
                         "home_s": home_time(state)})
            continue
        rows.append({"episode": e,
                     "time_s": round(float(g.timestamp.max()), 1),
                     "grip_cycles": grip_cycles(state[:, 5]),
                     "home_s": home_time(state),
                     "parcels": a.parcels or ""})
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "starts.jpg"), np.vstack(
        [np.hstack(tiles[j:j + 2]) for j in range(0, len(tiles), 2)]))
    pd.DataFrame(rows, columns=COLS).to_csv(old, index=False)
    print(f"{len(rows)} episodes -> {out}")


if __name__ == "__main__":
    main()
