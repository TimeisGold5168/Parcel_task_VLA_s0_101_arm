"""Offline start test: does a SmolVLA model leave the home pose?

At the start of each sort, the demos turn the wrist roll, open the
gripper, then reach. A model that lost this stays at home on the
robot. For a few home-pose frames, predict the next 1 s (3 seeds) and
print the median wrist roll turn, the share of plans that open the
gripper (> 20) and the share that move a reach joint by > 15 deg.

Frames: the old SmolVLA B run that left home (rollout_eval_parcel_
smolvla_b_medium), the DAgger run that stalled (..._smolvla_b_dg_
medium), and the dataset B episode starts.

Usage (model names of offline_eval_b.SPECS, optional :step):
  python -m parcel_vla.finetuning.start_test smolvla_b \
    smolvla_b_dgg:002000
"""

import sys

import numpy as np
import torch

from parcel_vla.evaluation import offline_eval as oe
from parcel_vla.evaluation import offline_eval_b as ob


def frames(m):
    new = ob.open_ds(m, "rollout_eval_parcel_smolvla_b_dg_medium")
    old = ob.open_ds(m, "rollout_eval_parcel_smolvla_b_medium")
    b = ob.open_ds(m, "parcel-sort-b")
    acts = oe.load_actions(ob.LOW / "parcel-sort-b")
    starts = [int(g.index[0]) for _, g in acts.groupby("episode_index")]
    return {
        "old run, start frame": [old[0]] * 4,
        "old run, 1.6 s": [old[48]] * 4,
        "stalled run, 30 s": [new[900]] * 4,
        "stalled run, 50 s": [new[1500]] * 4,
        "B starts (25 eps)": [b[i] for i in starts[::4]],
        "B starts + 1 s": [b[i + 30] for i in starts[::4]],
    }


def scores(m):
    """{frame label: (roll turn deg, gripper opens, reach > 15 deg)}"""
    out = {}
    for lab, items in frames(m).items():
        roll, grip, reach = [], [], []
        for seed in range(3):
            torch.manual_seed(seed)
            p = np.concatenate([
                m.predict(torch.utils.data.default_collate(
                    items[i:i + 8]))
                for i in range(0, len(items), 8)])
            # BSP plans are NaN after the segment: use the last valid
            # frame (a shorter turn than the 1 s of SmolVLA)
            ok = np.isfinite(p[:, :, 4])
            first = ok.argmax(1)
            last = p.shape[1] - 1 - ok[:, ::-1].argmax(1)
            r = np.arange(len(p))
            roll += list(np.abs(p[r, last, 4] - p[r, first, 4]))
            grip += list(np.nanmax(p[:, :, 5], 1))
            reach += list(np.nanmax(
                np.abs(p[:, :, :3] - p[:, :1, :3]), axis=1).max(1))
        out[lab] = (float(np.median(roll)),
                    float((np.array(grip) > 20).mean()),
                    float((np.array(reach) > 15).mean()))
    return out


def main():
    for arg in sys.argv[1:]:
        name, _, step = arg.partition(":")
        run, last = ob.SPECS[name]
        oe.MODELS[name] = ob.TRAIN / run
        m = oe.Model(name, step or last)
        print(f"== {arg}", flush=True)
        for lab, (roll, grip, reach) in scores(m).items():
            print(f"  {lab:22s} roll turn {roll:5.1f} deg | "
                  f"gripper opens {grip:4.2f} | "
                  f"reach > 15 deg {reach:4.2f}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
