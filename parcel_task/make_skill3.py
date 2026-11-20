"""Skill texts for the language test (one text per episode).

    python make_skill3.py [--variant base|distinct|distinct5]

base: 3 texts that are different in one word (box/bag). distinct: 3
texts with no shared word. distinct5: 5 texts with no shared word (also
far edge and on table). Works in place on Low101/parcel-sort-b-sk3,
-sk3d or -sk5d, each a copy of parcel-sort-b-all with hard-linked
videos. Episode ranges in that merge:
  0-98     parcel-sort-b (dataset B)  -> box or bag text
  99-114   far-edge corrections       -> far-edge text (distinct5)
  115-134  box-stack demos            -> top-box text
  135-164  stacked-parcel corrections -> top-box text
  165-187  black-box corrections      -> box text
  188-206  on-table corrections       -> on-table text (distinct5)
Episodes with no skill text keep the old text and are not trained.

Box or bag (dataset B): a box keeps the gripper open above 20 during
the carry (9-21 s), a bag lets it close in under 7 s (checked by eye on
96 of 99 episodes, all agree).

Writes outputs/train/skill3_split.json (skill3d_, skill5d_split.json):
texts, the box/bag tag of each dataset B episode, the held-out episodes
and the training episodes.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
TRAIN = Path(__file__).resolve().parents[2] / "train"
DISTINCT = {"box": "grasp rigid box with wide grip and set it at flip zone",
            "bag": "pinch soft bag by its edge then drop onto table",
            "top": "lift upper carton off stacked pile"}
# variant: (dataset, split file, texts)
VARIANTS = {
    "base": ("parcel-sort-b-sk3", "skill3_split.json",
             {"box": "pick the box and place it in the flip zone",
              "bag": "pick the bag and place it in the flip zone",
              "top": "pick the top box off the stack"}),
    "distinct": ("parcel-sort-b-sk3d", "skill3d_split.json", DISTINCT),
    "distinct5": ("parcel-sort-b-sk5d", "skill5d_split.json",
                  {**DISTINCT, "far": "reach far rim, pull parcel closer",
                   "push": "sweep lone item beyond wooden desk border"}),
}
B = range(0, 99)
FAR = range(99, 115)
STACK = range(115, 165)
BLACK_BOX = range(165, 188)
ON_TABLE = range(188, 207)
# held out: dataset B episodes with a box and a bag in the tray, and
# three of each other set
HELD = {"b_box": [40, 60, 80], "b_bag": [50, 70, 85],
        "stack": [117, 124, 131], "black_box": [167, 176, 185],
        "far_edge": [101, 107, 113], "on_table": [190, 197, 204]}
OPEN_BOX_S = 8.0


def b_kinds(root):
    data = pd.concat(pd.read_parquet(f) for f in
                     sorted(glob.glob(f"{root}/data/*/*.parquet")))
    kinds = {}
    for e in B:
        grip = np.stack(
            data[data.episode_index == e]["observation.state"].values)[:, 5]
        opened = grip > 20
        i0 = int(np.argmax(opened))
        shut = ~opened[i0:]
        i1 = i0 + (int(np.argmax(shut)) if shut.any() else len(shut))
        kinds[e] = "box" if (i1 - i0) / 30 > OPEN_BOX_S else "bag"
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ds_name, split, texts = VARIANTS[ap.parse_args().variant]
    root = LOW / ds_name
    out = TRAIN / split
    kinds = b_kinds(root)
    tasks = {e: texts[k] for e, k in kinds.items()}
    tasks.update({e: texts["top"] for e in STACK})
    tasks.update({e: texts["box"] for e in BLACK_BOX})
    held_out = {k: v for k, v in HELD.items()
                if k not in ("far_edge", "on_table")}
    if "far" in texts:
        tasks.update({e: texts["far"] for e in FAR})
        held_out["far_edge"] = HELD["far_edge"]
    if "push" in texts:
        tasks.update({e: texts["push"] for e in ON_TABLE})
        held_out["on_table"] = HELD["on_table"]
    for e, k in kinds.items():
        want = "box" if e in HELD["b_box"] else "bag" if e in \
            HELD["b_bag"] else k
        assert k == want, f"held-out episode {e} is a {k}"
    ds = LeRobotDataset("Low101/" + ds_name, root=root)
    modify_tasks(ds, episode_tasks=tasks)
    held = sorted(e for v in held_out.values() for e in v)
    train = sorted(e for e in tasks if e not in held)
    out.write_text(json.dumps({
        "dataset": ds_name, "texts": texts, "b_kind": kinds,
        "held_out": held_out, "train_episodes": train}, indent=1))
    n_box = sum(k == "box" for k in kinds.values())
    print(f"dataset B: {n_box} box, {len(kinds) - n_box} bag; "
          f"train {len(train)} episodes, held out {len(held)} -> {out}")


if __name__ == "__main__":
    main()
