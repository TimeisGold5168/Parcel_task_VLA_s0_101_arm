"""Offline fit test: does a model plan the grasp pose of its own demos?

For frames 1.0 s and 0.5 s before each grasp (gripper command below 30
after it was above 40), predict the next 1 s (3 seeds) and compare the
planned arm pose at the grasp frame with the demo action there:
shoulder lift, elbow, wrist flex (deg). Sets: the box-stack demos (in
the box model training data) and dataset B episodes.

  box model error on box demos ~ its error on B -> the demos are learned
    (more steps do not help; more demos may)
  box model error on box demos ~ old model error  -> underfit

  python outputs/portfolio/parcel_task/box_fit_test.py smolvla_b_box \
    smolvla_b
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import offline_eval as oe  # noqa: E402
import offline_eval_b as ob  # noqa: E402

GRIP_OPEN, GRIP_CLOSED = 40.0, 30.0
# sets cut with trim_corrections.py --grip-drop: grasp = command this
# far below its highest value (big boxes close at 30-46)
GRIP_DROP = {"rollout_dagger_far_stack_pull": 25.0}
JOINTS = [1, 2, 3]
LEADS = [30, 15]


def grasps(ds_name, every=1):
    root = ob.LOW / ds_name
    df = pd.concat([pd.read_parquet(f, columns=["index", "episode_index",
                                                "action"])
                    for f in sorted((root / "data").rglob("*.parquet"))])
    act = np.stack(df.action.values)
    out = []
    for e, g in df.groupby("episode_index"):
        if e % every:
            continue
        grip = np.stack(g.action.values)[:, 5]
        top = np.maximum.accumulate(grip)
        was_open = top > GRIP_OPEN
        if ds_name in GRIP_DROP:
            closed = grip <= top - GRIP_DROP[ds_name]
        else:
            closed = grip < GRIP_CLOSED
        c = np.flatnonzero(was_open & closed)
        if len(c) and c[0] >= max(LEADS):
            out.append(int(g["index"].values[c[0]]))
    return out, act


def fit(m, ds_name, every):
    ds = ob.open_ds(m, ds_name)
    gi, act = grasps(ds_name, every)
    res = {}
    for lead in LEADS:
        items = [ds[g - lead] for g in gi]
        truth = act[np.array(gi)][:, JOINTS]
        errs = []
        for seed in range(3):
            torch.manual_seed(seed)
            p = np.concatenate([
                m.predict(torch.utils.data.default_collate(items[i:i + 8]))
                for i in range(0, len(items), 8)])
            errs.append(p[:, lead - 1][:, JOINTS] - truth)
        res[lead] = np.mean(errs, 0)
    return len(gi), res


def main():
    sets = [("parcel-skill-boxstack-cut", 1), ("parcel-sort-b", 4)]
    for name in sys.argv[1:]:
        run, last = ob.SPECS[name]
        oe.MODELS[name] = ob.TRAIN / run
        m = oe.Model(name, last)
        print(f"== {name}", flush=True)
        for ds_name, every in sets:
            n, res = fit(m, ds_name, every)
            for lead, e in res.items():
                print(f"  {ds_name:26s} {lead / 30:.1f} s before grasp "
                      f"({n} eps): mean |err| {np.abs(e).mean():5.1f} deg"
                      f" | lift {e[:, 0].mean():+5.1f} elbow "
                      f"{e[:, 1].mean():+5.1f} flex {e[:, 2].mean():+5.1f}",
                      flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
