"""Collect top + side camera frames at the home pose, for the planner test.

A home visit: arm joints (shoulder pan, lift, elbow, wrist flex) within
HOME_TOL deg of the home pose for at least 0.5 s. One frame per visit,
0.5 s into the visit. Home pose: median first frame of dataset B.

  python -m parcel_vla.planner.collect_home_frames

Writes planner/frames/<id>_{top,side,wrist}.jpg and planner/frames/index.csv
(id, dataset, episode, frame, t). Labels go in planner/labels.csv.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from parcel_vla import paths

LOW = paths.LOW
OUT = Path(__file__).parent / "frames"
HOME_TOL = 8.0
MIN_VISIT = 15  # frames (0.5 s at 30 fps)
MIN_GAP = 90  # frames between two visits
# dataset: take every n-th episode start only (training data) or all
# home visits (robot runs)
SETS = {
    "parcel-sort-b": "starts:6",
    "rollout_eval_parcel_smolvla_b_fbs_hard": "visits",
    "rollout_eval_parcel_smolvla_b_fb_hard": "visits",
    "rollout_eval_parcel_smolvla_b_hard": "visits",
    "rollout_eval_parcel_smolvla_b_medium": "visits",
    "rollout_eval_parcel_smolvla_b_medium_old": "visits",
    "rollout_eval_parcel_smolvla_b_box_medium": "visits",
    "rollout_eval_parcel_smolvla_b_easy": "visits",
    "rollout_eval_parcel_diffusion_b_easy": "visits",
    "rollout_eval_parcel_smolvla_b_fe_hard": "visits",
}


def load(ds):
    root = LOW / ds
    df = pd.concat(pd.read_parquet(f, columns=["episode_index",
                                                "frame_index",
                                                "observation.state"])
                   for f in sorted((root / "data").rglob("*.parquet")))
    eps = pd.concat(pd.read_parquet(f)
                    for f in sorted((root / "meta/episodes")
                                    .rglob("*.parquet")))
    return df, eps


def home_pose():
    df, _ = load("parcel-sort-b")
    first = df.groupby("episode_index").head(1)
    return np.median(np.stack(first["observation.state"].values), 0)


def visits(state, home):
    at = np.abs(state[:, :4] - home[:4]).max(1) < HOME_TOL
    out, start, last = [], None, -MIN_GAP
    for i, v in enumerate(np.r_[at, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_VISIT and start - last >= MIN_GAP:
                out.append(start + MIN_VISIT)
                last = start
            start = None
    return out


def grab(ds, ep_row, cam, t, path):
    k = f"videos/observation.images.{cam}"
    f = (LOW / ds / f"videos/observation.images.{cam}"
         / f"chunk-{int(ep_row[k + '/chunk_index']):03d}"
         / f"file-{int(ep_row[k + '/file_index']):03d}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss",
                    str(ep_row[k + "/from_timestamp"] + t), "-i", str(f),
                    "-frames:v", "1", "-q:v", "3", str(path)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    home = home_pose()
    rows = []
    for ds, mode in SETS.items():
        df, eps = load(ds)
        for e, g in df.groupby("episode_index"):
            if mode.startswith("starts"):
                if e % int(mode.split(":")[1]):
                    continue
                idx = [5]
            else:
                idx = visits(np.stack(g["observation.state"].values), home)
            er = eps[eps.episode_index == e].iloc[0]
            for fi in idx:
                fid = f"{len(rows):03d}"
                t = fi / 30.0
                for cam in ("top", "side", "wrist"):
                    grab(ds, er, cam, t, OUT / f"{fid}_{cam}.jpg")
                rows.append({"id": fid, "dataset": ds, "episode": int(e),
                             "frame": int(fi), "t": round(t, 1)})
        print(ds, sum(r["dataset"] == ds for r in rows), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)
    print(len(rows), "frames")


if __name__ == "__main__":
    main()
