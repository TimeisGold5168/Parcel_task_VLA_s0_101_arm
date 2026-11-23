"""Planner test: box or bag from the wrist camera during the approach.

The policy selects its target from the image (robot test A). Thus the
planner can wait until the arm moves to a parcel, then read the kind
of that parcel in the wrist camera, and send the skill text one time.
Frames: dataset B, 1.0 s and 0.5 s before the grasp (the gripper
closes 10 below its highest value after it opens above 25, on the
action). Labels: the box/bag tag of each episode
(outputs/train/skill3_split.json).

  python -m parcel_vla.planner.wrist_kind

Prints the zero-shot and probe accuracy for each model and offset.
"""

import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from parcel_vla.planner.next_kind import HEAP, MODELS, SPLIT, ask, probe_cv
from parcel_vla.planner import next_kind as nk
from parcel_vla.planner.probe import auc
from parcel_vla.planner.vlm_planner import VLMPlanner

from parcel_vla.benchmark import bench_sheets as bs
from parcel_vla.skill_texts.lang_test import grasp_frame

OFFSETS = (30, 15)  # frames before the grasp
QUESTION = ("Look at the parcel in front of the robot gripper. Is it a "
            "soft plastic mailer bag?")


def main():
    nk.QUESTION = QUESTION
    split = json.loads(SPLIT.read_text())
    root = bs.LOW / "parcel-sort-b"
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted((root / "meta/episodes").rglob("*.parquet")))
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pd.concat(pd.read_parquet(f, columns=["episode_index", "action"])
                     for f in sorted((root / "data").rglob("*.parquet")))
    images = {o: [] for o in OFFSETS}
    for _, r in eps.iterrows():
        act = np.stack(data[data.episode_index == r.episode_index]
                       .action.values)
        g = grasp_frame(act[:, 5])
        ts = [max(g - o, 0) / 30 for o in OFFSETS]
        for o, bgr in zip(OFFSETS, bs.frames_at(root, r, "wrist", ts)):
            images[o].append(
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    y = np.array([split["b_kind"][str(int(e))] == "bag"
                  for e in eps.episode_index], float)
    groups = eps.episode_index.values // HEAP
    for model_id in MODELS:
        name = model_id.split("/")[-1]
        planner = VLMPlanner(model_id)
        for o in OFFSETS:
            p_zero, feats, secs = [], [], []
            for im in images[o]:
                t0 = time.perf_counter()
                p, feat = ask(planner, [im])
                secs.append(time.perf_counter() - t0)
                p_zero.append(p)
                feats.append(feat)
            p_zero = np.array(p_zero)
            p_probe = probe_cv(np.stack(feats), y, groups)
            for kind, p in (("zero-shot", p_zero), ("probe", p_probe)):
                acc = ((p >= 0.5) == (y == 1)).mean()
                print(f"{name:28s} {o / 30:.1f} s before grasp {kind:9s} "
                      f"accuracy {acc * 100:4.0f}%  AUC {auc(y, p):.2f}  "
                      f"{np.mean(secs) * 1000:.0f} ms", flush=True)
        del planner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
