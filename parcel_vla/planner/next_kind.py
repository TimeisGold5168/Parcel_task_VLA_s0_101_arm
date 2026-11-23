"""Planner test: the kind of the parcel that the policy picks next.

The policy goes to the parcel nearest to the flip zone (robot test A).
At home, the planner must say if that parcel is a box or a bag, and
send that skill text. Labels: the dataset B episode starts (arm at
home); the operator then took that parcel, and its box/bag tag is in
outputs/train/skill3_split.json (gripper rule, checked by eye).

For each model, one forward pass per frame with QUESTION:
  zero-shot  P(yes) from the "Yes"/"No" logits (yes = bag)
  probe      logistic regression on the last hidden state, 5-fold
             cross-validation by heap (5 episodes per heap)

  python -m parcel_vla.planner.next_kind

Writes runs/planner/next_kind.csv and prints the accuracy and the time.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from parcel_vla import paths
from parcel_vla.benchmark import bench_sheets as bs
from parcel_vla.planner.probe import FOLDS, WRIST_CROP, auc, logreg
from parcel_vla.planner.vlm_planner import (SIDE_CROP, TOP_CROP, VLMPlanner,
                                            crop)

HERE = Path(__file__).parent
SPLIT = paths.TRAIN / "skill3_split.json"
MODELS = ["HuggingFaceTB/SmolVLM2-2.2B-Instruct",
          "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
QUESTION = ("Look at the parcels on the yellow tray. Find the parcel that "
            "is nearest to the wooden table on the right. Is that parcel "
            "a soft plastic mailer bag?")
HEAP = 5
START_S = 0.3


def start_images(root, row):
    out = []
    for cam, box in (("top", TOP_CROP), ("side", SIDE_CROP),
                     ("wrist", WRIST_CROP)):
        bgr = bs.frames_at(root, row, cam, [START_S])[0]
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out.append(crop(img, box))
    return out


@torch.no_grad()
def ask(planner, images):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text",
                    "text": QUESTION + " Answer yes or no."})
    inp = planner.proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt").to(planner.device)
    inp["pixel_values"] = inp["pixel_values"].to(planner.dtype)
    out = planner.model(**inp, output_hidden_states=True)
    logits = out.logits[0, -1].float()
    y = torch.logsumexp(logits[list(planner.yes)], 0)
    n = torch.logsumexp(logits[list(planner.no)], 0)
    feat = out.hidden_states[-1][0, -1].float().cpu().numpy()
    return float(torch.sigmoid(y - n)), feat


def probe_cv(feats, y, groups):
    x = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    order = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    fold = {g: i % FOLDS for i, g in enumerate(order)}
    f = np.array([fold[g] for g in groups])
    p = np.zeros(len(y))
    for i in range(FOLDS):
        tr, te = f != i, f == i
        w, b = logreg(x[tr], y[tr])
        p[te] = 1 / (1 + np.exp(-(x[te] @ w + b)))
    return p


def main():
    split = json.loads(SPLIT.read_text())
    root = bs.LOW / "parcel-sort-b"
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted((root / "meta/episodes").rglob("*.parquet")))
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    images = [start_images(root, r) for _, r in eps.iterrows()]
    y = np.array([split["b_kind"][str(int(e))] == "bag"
                  for e in eps.episode_index], float)
    groups = eps.episode_index.values // HEAP
    res = pd.DataFrame({"episode": eps.episode_index,
                        "kind": np.where(y == 1, "bag", "box")})
    print(f"{len(y)} start frames: {int(y.sum())} bag, "
          f"{int(len(y) - y.sum())} box")
    for model_id in MODELS:
        name = model_id.split("/")[-1]
        planner = VLMPlanner(model_id)
        ask(planner, images[0])
        p_zero, feats, secs = [], [], []
        for ims in images:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p, feat = ask(planner, ims)
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
            p_zero.append(p)
            feats.append(feat)
        mem = torch.cuda.max_memory_allocated() / 2**30
        del planner
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        p_zero = np.array(p_zero)
        p_probe = probe_cv(np.stack(feats), y, groups)
        res[f"zero_{name}"] = p_zero.round(3)
        res[f"probe_{name}"] = p_probe.round(3)
        for kind, p in (("zero-shot", p_zero), ("probe", p_probe)):
            acc = ((p >= 0.5) == (y == 1)).mean()
            print(f"{name:28s} {kind:9s} accuracy {acc * 100:4.0f}%  "
                  f"AUC {auc(y, p):.2f}")
        print(f"{name:28s} {np.mean(secs) * 1000:.0f} ms per decision, "
              f"GPU peak {mem:.1f} GB")
    out = paths.RUNS / "planner"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "next_kind.csv", index=False)


if __name__ == "__main__":
    main()
