"""Train the two box/bag probes of the live planner (SmolVLM2-500M).

  home   at the home pose (top + side + wrist crops): the kind of the
         parcel that the policy goes to next (next_kind.py: 74 %)
  wrist  during the approach (wrist image, 1 s before the grasp): the
         kind of the parcel in front of the gripper (wrist_kind.py:
         99 %)

Labels: dataset B, 99 episodes (box/bag tag of make_skill3.py). Each
probe is a logistic regression on the last hidden state, trained on
all 99 frames.

  python -m parcel_vla.planner.train_planner

Writes parcel_vla/planner/planner_500m.npz (read by live_planner.py).
"""

import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from parcel_vla.planner import next_kind as nk
from parcel_vla.planner.probe import logreg
from parcel_vla.planner.vlm_planner import VLMPlanner
from parcel_vla.planner.wrist_kind import QUESTION as WRIST_Q

from parcel_vla.benchmark import bench_sheets as bs
from parcel_vla.skill_texts.lang_test import grasp_frame

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OUT = nk.HERE / "planner_500m.npz"
WRIST_LEAD = 30  # frames before the grasp
HOME_Q = nk.QUESTION


def fit(feats, y):
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    w, b = logreg((feats - mu) / sd, y)
    p = 1 / (1 + np.exp(-(((feats - mu) / sd) @ w + b)))
    return mu, sd, w, b, ((p >= 0.5) == (y == 1)).mean()


def main():
    split = json.loads(nk.SPLIT.read_text())
    root = bs.LOW / "parcel-sort-b"
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted((root / "meta/episodes").rglob("*.parquet")))
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pd.concat(pd.read_parquet(f, columns=["episode_index", "action"])
                     for f in sorted((root / "data").rglob("*.parquet")))
    y = np.array([split["b_kind"][str(int(e))] == "bag"
                  for e in eps.episode_index], float)
    planner = VLMPlanner(MODEL)
    home, wrist = [], []
    for _, r in eps.iterrows():
        nk.QUESTION = HOME_Q
        home.append(nk.ask(planner, nk.start_images(root, r))[1])
        act = np.stack(data[data.episode_index == r.episode_index]
                       .action.values)
        t = max(grasp_frame(act[:, 5]) - WRIST_LEAD, 0) / 30
        bgr = bs.frames_at(root, r, "wrist", [t])[0]
        nk.QUESTION = WRIST_Q
        wrist.append(nk.ask(planner, [Image.fromarray(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))])[1])
    out = {"model": MODEL, "home_question": HOME_Q,
           "wrist_question": WRIST_Q}
    for name, feats in (("home", np.stack(home)), ("wrist", np.stack(wrist))):
        mu, sd, w, b, acc = fit(feats, y)
        out.update({f"{name}_mu": mu, f"{name}_sd": sd, f"{name}_w": w,
                    f"{name}_b": b})
        print(f"{name}: training accuracy {acc * 100:.0f}% "
              f"({len(y)} frames)")
    np.savez(OUT, **out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
