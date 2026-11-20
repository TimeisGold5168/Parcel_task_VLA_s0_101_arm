"""Try image and prompt variants for the stack question (AUC per variant).

  python outputs/portfolio/planner/stack_variants.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vlm_planner import SIDE_CROP, TOP_CROP, VLMPlanner, crop

HERE = Path(__file__).parent
PROMPTS = {
    "orig": "Look at the parcels on the yellow tray. Is one box stacked "
            "on top of another box?",
    "side": "This is a side view of a yellow tray with parcels. Is a box "
            "sitting on top of another box, so that two boxes form a "
            "tower?",
    "count": "Are there two boxes placed one above the other, the upper "
             "box resting on the lower box?",
}
# variant: (images, prompt)
VARIANTS = {
    "top+side orig": ("ts", "orig"),
    "wrist orig": ("w", "orig"),
    "top+side+wrist orig": ("tsw", "orig"),
}
WRIST_CROP = (0.0, 0.0, 1.0, 0.5)


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def run(planner, lab):
    out = {}
    for name, (imgs, pk) in VARIANTS.items():
        ps, t0 = [], time.perf_counter()
        for _, r in lab.iterrows():
            top = crop(Image.open(HERE / "frames" / f"{r.id}_top.jpg"),
                       TOP_CROP)
            side = crop(Image.open(HERE / "frames" / f"{r.id}_side.jpg"),
                        SIDE_CROP)
            wrist = crop(Image.open(HERE / "frames" / f"{r.id}_wrist.jpg"),
                         WRIST_CROP)
            ims = [{"t": top, "s": side, "w": wrist}[c] for c in imgs]
            ps.append(planner.p_yes(ims, PROMPTS[pk]))
        p, y = np.array(ps), lab["stack"].values
        dt = (time.perf_counter() - t0) / len(lab)
        best = max(np.unique(p), key=lambda t: ((p >= t) == (y == 1)).mean())
        acc = ((p >= best) == (y == 1)).mean()
        rec = ((p >= best) & (y == 1)).sum() / (y == 1).sum()
        out[name] = (auc(y, p), best, acc, rec, dt)
        print(f"  {name:16s} AUC {out[name][0]:.2f}  thr {best:.3f} "
              f"acc {acc * 100:.0f}% stacks found {rec * 100:.0f}% "
              f"{dt:.2f} s", flush=True)
    return out


def main():
    lab = pd.read_csv(HERE / "labels.csv", dtype={"id": str})
    lab = lab[(lab.skip == 0) & lab["stack"].notna()]
    for split in (False, True):
        print(f"== image tiles: {split}", flush=True)
        planner = VLMPlanner("HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                             split=split)
        run(planner, lab)
        del planner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
