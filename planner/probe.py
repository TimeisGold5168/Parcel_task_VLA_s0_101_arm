"""Linear probe on the VLM features: learn our labels from few frames.

The VLM reads the question with the three camera crops (no tiles). The
hidden state of the last token is the feature. A logistic regression
per question learns our label definitions. Test: cross-validation by
episode (frames of one episode are never in train and test at once).

  python outputs/portfolio/planner/probe.py \
    [--model HuggingFaceTB/SmolVLM2-500M-Video-Instruct --device cpu]

Features are cached per model (probe_features.npz for the 2.2B model,
probe_features_<model>.npz for others); results in probe_cv*.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vlm_planner import QUESTIONS, SIDE_CROP, TOP_CROP, VLMPlanner, crop

HERE = Path(__file__).parent
WRIST_CROP = (0.0, 0.0, 1.0, 0.5)
FOLDS = 5
BASE = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


@torch.no_grad()
def feature(planner, images, question):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text",
                    "text": question + " Answer yes or no."})
    inp = planner.proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt").to(planner.device)
    inp["pixel_values"] = inp["pixel_values"].to(planner.dtype)
    out = planner.model(**inp, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().cpu().numpy()


def logreg(x, y, l2=1.0, steps=500, lr=0.1):
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        w -= lr * (x.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * (p - y).mean()
    return w, b


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    suffix = "" if a.model == BASE else "_" + a.model.split("/")[-1]
    lab = pd.read_csv(HERE / "labels.csv", dtype={"id": str})
    idx = pd.read_csv(HERE / "frames/index.csv", dtype={"id": str})
    lab = lab[lab.skip == 0].merge(idx, on="id")
    lab["group"] = lab.dataset + ":" + lab.episode.astype(str)
    cache = HERE / f"probe_features{suffix}.npz"
    if cache.exists():
        feats = dict(np.load(cache))
    else:
        # bfloat16 is slow on most CPUs
        dtype = torch.float32 if a.device == "cpu" else torch.bfloat16
        planner = VLMPlanner(a.model, device=a.device, dtype=dtype)
        feats = {k: [] for k in QUESTIONS}
        for n, (_, r) in enumerate(lab.iterrows()):
            ims = [crop(Image.open(HERE / "frames" / f"{r.id}_{c}.jpg"), b)
                   for c, b in (("top", TOP_CROP), ("side", SIDE_CROP),
                                ("wrist", WRIST_CROP))]
            for k, q in QUESTIONS.items():
                feats[k].append(feature(planner, ims, q))
            print(f"features {n + 1}/{len(lab)}", flush=True)
        feats = {k: np.stack(v) for k, v in feats.items()}
        np.savez(cache, **feats)
    groups = lab.group.unique()
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    fold = {g: i % FOLDS for i, g in enumerate(groups)}
    lab["fold"] = lab.group.map(fold)
    print(f"{a.model}: {len(lab)} frames, {len(groups)} episodes, "
          f"{FOLDS} folds")
    for k in QUESTIONS:
        ok = lab[k].notna().values
        x, y, f = feats[k][ok], lab.loc[ok, k].values.astype(float), \
            lab.loc[ok, "fold"].values
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        p = np.zeros(len(y))
        for i in range(FOLDS):
            tr, te = f != i, f == i
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                p[te] = y[tr].mean()
                continue
            w, b = logreg(x[tr], y[tr])
            p[te] = 1 / (1 + np.exp(-(x[te] @ w + b)))
        acc = ((p >= 0.5) == (y == 1)).mean()
        rec = ((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)
        false = ((p >= 0.5) & (y == 0)).sum()
        a_ = auc(y, p) if 0 < y.sum() < len(y) else float("nan")
        print(f"  {k:10s} AUC {a_:.2f} acc {acc * 100:.0f}% "
              f"yes found {rec * 100:.0f}% ({int(y.sum())}/{len(y)} yes), "
              f"false yes {false}")
        lab.loc[ok, f"cv_{k}"] = p
    lab.to_csv(HERE / f"probe_cv{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
