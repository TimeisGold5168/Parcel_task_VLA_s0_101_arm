"""Cut the idle home-pose tail from the end of each DAgger correction.

Each correction ends with the arm back at the home pose, still. These
frames teach "at home: stay", but dataset B has "at home: leave" at the
start of each episode. With the tails, the home frames in the merged
data are 54 % "stay" and the fine-tuned models do not leave home.

Each episode is cut after its last frame away from the home pose
(home = shoulder lift < -100 deg and elbow > 85 deg). The videos are
not changed (hard links): an episode points to its own time range in
the video files. The data rows, the global index, the episode metadata
and the stats (non-image features) are written again.

--mode pull (far-edge corrections): keep each episode only up to the
pick of the parcel out of the far zone: the grasp (gripper command
below 30 after it was above 40; with --grip-drop D: D below its highest
value, for big boxes), then the first frame with shoulder lift
< 20 deg, plus --margin frames (default 15), and never past the flip
onset (wrist roll 20 deg away from its grasp value). Flip, exit push
and return are the same as in dataset B, so they are cut.
A correction that starts at the home pose also loses its frames at
home: it starts 5 frames before the first motion (as --mode start).

--mode start (skill demos): cut the idle wait at the start of each
episode (hands still in the scene) to --margin frames (default 15)
before the first motion (a joint 5 deg away from its first value).
Frame index and timestamp start again at 0; the video start moves.

Usage:
  python -m parcel_vla.finetuning.trim_corrections \
    --src rollout_dagger_parcel_b_noint --dst rollout_dagger_parcel_b_trim
  python -m parcel_vla.finetuning.trim_corrections --mode pull \
    --src rollout_dagger_far_edge --dst rollout_dagger_far_edge_pull
  python -m parcel_vla.finetuning.trim_corrections --mode start \
    --src parcel-skill-boxstack --dst parcel-skill-boxstack-cut
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from parcel_vla import paths
from lerobot.datasets.compute_stats import (aggregate_stats,
                                            compute_episode_stats)
from lerobot.datasets.io_utils import write_stats

LOW = paths.LOW
LIFT_HOME = -100.0
ELBOW_HOME = 85.0
LIFT_PULLED = 20.0
GRIP_OPEN = 40.0
GRIP_CLOSED = 30.0
ROLL_FLIP = 20.0
MOVE = 5.0


def keep_len(state, margin):
    away = ~((state[:, 1] < LIFT_HOME) & (state[:, 2] > ELBOW_HOME))
    out = np.flatnonzero(away)
    if len(out) == 0:
        return len(state)
    return min(len(state), int(out[-1]) + 1 + margin)


def pull_len(state, action, margin, grip_drop=None):
    grip = action[:, 5]
    top = np.maximum.accumulate(grip)
    was_open = top > GRIP_OPEN
    if grip_drop is None:
        closed = np.flatnonzero(was_open & (grip < GRIP_CLOSED))
    else:
        closed = np.flatnonzero(was_open & (grip <= top - grip_drop))
    if len(closed) == 0:
        return len(state)
    g = int(closed[0])
    pulled = np.flatnonzero(state[g:, 1] < LIFT_PULLED)
    flip = np.flatnonzero(np.abs(state[g:, 4] - state[g, 4]) > ROLL_FLIP)
    end = g + int(pulled[0]) + margin if len(pulled) else len(state)
    if len(flip):
        end = min(end, g + int(flip[0]))
    return min(len(state), end + 1)


def start_cut(state, margin):
    moved = np.abs(state[:, :5] - state[0, :5]).max(1) > MOVE
    i = np.flatnonzero(moved)
    return max(0, int(i[0]) - margin) if len(i) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="rollout_dagger_parcel_b_noint")
    ap.add_argument("--dst", default="rollout_dagger_parcel_b_trim")
    ap.add_argument("--mode", choices=["home", "pull", "start"],
                    default="home")
    ap.add_argument("--margin", type=int, default=None,
                    help="frames kept after the cut point, or before "
                    "it for start (default: 0 for home, else 15)")
    ap.add_argument("--grip-drop", type=float, default=None,
                    help="pull: grasp = command this far below its max")
    args = ap.parse_args()
    if args.margin is None:
        args.margin = 0 if args.mode == "home" else 15
    src, dst = LOW / args.src, LOW / args.dst
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)
    shutil.copytree(src / "meta", dst / "meta")

    info = json.loads((src / "meta/info.json").read_text())
    fps = info["fps"]
    feats = {k: v for k, v in info["features"].items()
             if v["dtype"] not in ("video", "image")}

    files = sorted((src / "data").rglob("*.parquet"))
    data = {f: pd.read_parquet(f) for f in files}
    ep_files = sorted((dst / "meta/episodes").rglob("*.parquet"))
    eps = {f: pd.read_parquet(f) for f in ep_files}

    keep, first = {}, {}
    for df in data.values():
        for e, g in df.groupby("episode_index"):
            st = np.stack(g["observation.state"].values)
            first[int(e)] = 0
            if args.mode == "start":
                first[int(e)] = start_cut(st, args.margin)
                keep[int(e)] = len(st) - first[int(e)]
                print(f"  episode {int(e)}: cut {first[int(e)] / fps:.1f}"
                      f" s at the start")
            elif args.mode == "pull":
                act = np.stack(g["action"].values)
                end = pull_len(st, act, args.margin, args.grip_drop)
                if st[0, 1] < LIFT_HOME and st[0, 2] > ELBOW_HOME:
                    first[int(e)] = min(start_cut(st, 5), end - 1)
                keep[int(e)] = end - first[int(e)]
                print(f"  episode {int(e)}: keep "
                      f"{first[int(e)] / fps:.1f}-{end / fps:.1f}"
                      f" of {len(st) / fps:.1f} s")
            else:
                keep[int(e)] = keep_len(st, args.margin)

    # rows, global index (dense, in episode order)
    nxt, start = 0, {}
    for e in sorted(keep):
        start[e] = nxt
        nxt += keep[e]
    ep_stats = {}
    for f, df in data.items():
        f0 = df.episode_index.map(first)
        df = df[(df.frame_index >= f0)
                & (df.frame_index < f0 + df.episode_index.map(keep))].copy()
        f0 = df.episode_index.map(first)
        df["frame_index"] = df.frame_index - f0
        df["timestamp"] = (df.frame_index / fps).astype(df.timestamp.dtype)
        df["index"] = df.episode_index.map(start) + df.frame_index
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        for e, g in df.groupby("episode_index"):
            ep = {k: (np.stack(g[k].values) if g[k].dtype == object
                      else g[k].values) for k in feats}
            ep_stats[int(e)] = compute_episode_stats(ep, feats)

    all_stats = []
    for f, em in eps.items():
        e = em.episode_index.astype(int)
        n = e.map(keep)
        em["length"] = n
        em["dataset_from_index"] = e.map(start)
        em["dataset_to_index"] = e.map(start) + n
        for c in em.columns:
            if c.endswith("/from_timestamp"):
                em[c] = em[c] + e.map(first) / fps
        for c in em.columns:
            if c.endswith("/to_timestamp"):
                base = c.replace("/to_timestamp", "/from_timestamp")
                em[c] = em[base] + n / fps
        for c in [c for c in em.columns if c.startswith("stats/")]:
            _, k, name = c.split("/", 2)
            if k in feats:
                dt = np.asarray(em[c].iloc[0]).dtype
                em[c] = [np.asarray(ep_stats[int(x)][k][name], dt)
                         .reshape(-1) for x in e]
        for _, row in em.iterrows():
            st = {}
            for c in em.columns:
                if c.startswith("stats/"):
                    _, k, name = c.split("/", 2)
                    v = row[c]
                    v = np.asarray(v)
                    if v.dtype == object:  # image stats: (3,) of (1,)(1,)
                        v = np.stack([np.stack(x) for x in v])
                        v = v.reshape(-1, 1, 1)
                    st.setdefault(k, {})[name] = v
            all_stats.append(st)
        em.to_parquet(f, index=False)

    write_stats(aggregate_stats(all_stats), dst)
    before = info["total_frames"]
    info["total_frames"] = nxt
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"{args.src} -> {args.dst}: {before} -> {nxt} frames "
          f"({before - nxt} cut, {len(keep)} episodes)")


if __name__ == "__main__":
    main()
