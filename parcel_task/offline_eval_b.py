"""Offline evaluation of the dataset B models (no robot).

Dataset B (Low101/parcel-sort-b): pick the parcel from the stack, place
it in the flip zone, flip it if the label is up, release it label down
(mark), push it off the table. All 99 episodes are in training.

Held-out episodes (not in training): parcel-full-new-pilot 0-4 (same
procedure, recorded before B) and parcel-sort-b_old episode 88 (removed
from B: a second grasp after the release). Both datasets were deleted
on 2026-11-18; a held-out set that is not on disk is skipped.

Kind of each demo: "flip" if the wrist roll moves more than 135 deg
before the mark (B: 75-116 deg without a flip, 159-220 deg with one).
Grasp = the first gripper close after the first open. Flip onset = the
first frame after the grasp with the wrist roll 20 deg away from its
grasp value. No-flip episodes use the median onset point (0.4 of the
grasp-to-mark time).

Test 1 (action error): every 15 frames, predict the next 1 s (30
frames) and compare with the demo. Mean absolute error in degrees (5
arm joints; gripper apart). Phases: approach (to the grasp), carry
(grasp to mark: place, flip, release), exit (after the mark).

Test 2 (flip decision): every 10 frames from the grasp to 10 frames
before the flip onset, predict the next 1 s. Score = the largest wrist
roll range in a predicted second. Decision "flip" if the score is more
than 30 deg. The same score on the demo futures is the probe (95/99 on
B). The images then show the parcel grasped, but not yet turned.

Test 3 (image swap): at onset - 10 frames, keep the arm state, swap in
the camera images of an episode of the other kind at its own onset - 10
frames. "Follows the images" = the decision changes to the image kind.

Usage:
  python outputs/portfolio/parcel_task/offline_eval_b.py
  python outputs/portfolio/parcel_task/offline_eval_b.py --models smolvla_b
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import offline_eval as oe  # noqa: E402

LOW = oe.LOW
TRAIN = oe.TRAIN
# name: (run folder, checkpoint step)
SPECS = {
    "smolvla_b": ("smolvla_parcel_b_80k", "last"),
    "smolvla_b40k": ("smolvla_parcel_b_80k", "040000"),
    "smolvla_bsp_b": ("smolvla_bsp_parcel_b_40k", "last"),
    "diffusion_b": ("diffusion_parcel_b_80k", "last"),
    "diffusion_bsp_b": ("diffusion_bsp_parcel_b_40k", "last"),
    "smolvla_b_dg": ("smolvla_parcel_b_dagger_15k", "last"),
    "smolvla_bsp_b_dg": ("smolvla_bsp_parcel_b_dagger_15k", "last"),
    "smolvla_b_dgt": ("smolvla_parcel_b_dagger_trim_15k", "last"),
    "smolvla_b_dgg": ("smolvla_parcel_b_dagger_gentle_6k", "last"),
    "smolvla_b_ctl": ("smolvla_parcel_b_control_10k", "last"),
    "smolvla_b_fe": ("smolvla_parcel_b_fe_5k", "last"),
    "smolvla_b_box": ("smolvla_parcel_b_box_5k", "last"),
    "smolvla_b_fb": ("smolvla_parcel_b_fb_15k", "last"),
    "smolvla_b_fe15": ("smolvla_parcel_b_fe_15k", "last"),
    "smolvla_b_fbs": ("smolvla_parcel_b_fbs_10k", "last"),
    "smolvla_bsp_b_fbs": ("smolvla_bsp_parcel_b_fbs_30k", "last"),
    "smolvla_bsp_b_fb": ("smolvla_bsp_parcel_b_fb_30k", "last"),
    "smolvla_b_all": ("smolvla_parcel_b_all_20k", "last"),
    "smolvla_b_allb": ("smolvla_parcel_b_allb_20k", "last"),
    "smolvla_bsp_b_all": ("smolvla_bsp_parcel_b_all_30k", "last"),
}
HELD_OUT = [p for p in [("parcel-full-new-pilot", [0, 1, 2, 3, 4]),
                        ("parcel-sort-b_old", [88])]
            if (Path.home() / ".cache/huggingface/lerobot/Low101"
                / p[0] / "meta/info.json").exists()]
TRAIN_REF = [("parcel-sort-b", [3, 17, 31, 45, 59, 73, 87])]
TRAIN_DEC = [("parcel-sort-b", list(range(99)))]
H = 30
FLIP_ROLL = 135.0  # deg of wrist roll before the mark
ONSET_ROLL = 20.0
ONSET_FRAC = 0.4
DEC_GAP = 10  # last decision sample, frames before the onset
DEC_EVERY = 10
DEC_T = 30.0  # deg of wrist roll in one predicted second


def episodes_info(ds_name):
    """Per episode: start index, length, grasp, mark, kind, onset."""
    root = LOW / ds_name
    acts = oe.load_actions(root)
    marks = pd.read_csv(root / "phase_marks.csv")
    act = np.stack(acts.action.values)
    info = {}
    for e, g in acts.groupby("episode_index"):
        i0, n = int(g.index[0]), len(g)
        a = act[i0:i0 + n]
        mk = int(marks[marks.episode == e].frame.values[0])
        o = int(np.argmax(a[:, 5] > 25))
        hi, grasp = a[o, 5], mk
        for i in range(o, mk):
            hi = max(hi, a[i, 5])
            if a[i, 5] < hi - 10:
                grasp = i
                break
        kind = "flip" if np.ptp(a[:mk, 4]) > FLIP_ROLL else "noflip"
        if kind == "flip":
            d = np.abs(a[grasp:mk, 4] - a[grasp, 4])
            onset = grasp + int(np.argmax(d > ONSET_ROLL))
        else:
            onset = grasp + int(ONSET_FRAC * (mk - grasp))
        info[e] = dict(start=i0, n=n, grasp=grasp, mark=mk, kind=kind,
                       onset=onset)
    return act, info


def phase_of(f, ep):
    if f < ep["grasp"]:
        return "approach"
    return "carry" if f < ep["mark"] else "exit"


def open_ds(model, ds_name):
    root = LOW / ds_name
    d = model.delta(30)
    delta = None
    if d is not None:
        feats = json.loads((root / "meta/info.json").read_text())["features"]
        delta = {k: d for k in feats if k.startswith("observation")}
    return LeRobotDataset("Low101/" + ds_name, root=root,
                          delta_timestamps=delta)


def predict_rows(model, ds, idx, batch_size, items=None):
    preds = []
    for i in range(0, len(idx), batch_size):
        chunk = (items[i:i + batch_size] if items is not None
                 else [ds[j] for j in idx[i:i + batch_size]])
        torch.manual_seed(i)
        preds.append(model.predict(torch.utils.data.default_collate(chunk)))
    return np.concatenate(preds)


def roll_score(fut):
    """Largest wrist roll range in a predicted second (NaN-safe)."""
    r = fut[:, 4]
    r = r[~np.isnan(r)]
    return float(np.ptp(r)) if len(r) > 1 else 0.0


def test1(model, parts, set_name, every, batch_size):
    rows = []
    for ds_name, eps in parts:
        act, info = episodes_info(ds_name)
        ds = open_ds(model, ds_name)
        idx, meta = [], []
        for e in eps:
            ep = info[e]
            for f in range(0, ep["n"] - H, every):
                idx.append(ep["start"] + f)
                meta.append((ds_name, e, f, phase_of(f, ep)))
        pred = predict_rows(model, ds, idx, batch_size)
        gt = np.stack([act[j:j + H] for j in idx])
        err = np.abs(pred - gt)
        df = pd.DataFrame(meta, columns=["dataset", "episode", "frame",
                                         "phase"])
        df["arm_mae"] = np.nanmean(err[:, :, :5], axis=(1, 2))
        df["grip_mae"] = np.nanmean(err[:, :, 5], axis=1)
        df["set"] = set_name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def dec_frames(ep):
    last = max(ep["onset"] - DEC_GAP, ep["grasp"] + 1)
    return list(range(ep["grasp"], last, DEC_EVERY)) or [ep["grasp"]]


def test2(model, parts, set_name, batch_size):
    rows = []
    for ds_name, eps in parts:
        act, info = episodes_info(ds_name)
        ds = open_ds(model, ds_name) if model is not None else None
        idx, owner = [], []
        for e in eps:
            for f in dec_frames(info[e]):
                idx.append(info[e]["start"] + f)
                owner.append(e)
        pred = (predict_rows(model, ds, idx, batch_size)
                if model is not None else None)
        for e in eps:
            k = [i for i, o in enumerate(owner) if o == e]
            demo = max(roll_score(act[idx[i]:idx[i] + H]) for i in k)
            row = dict(set=set_name, dataset=ds_name, episode=e,
                       truth=info[e]["kind"], probe_score=demo,
                       probe_kind="flip" if demo > DEC_T else "noflip")
            if pred is not None:
                s = max(roll_score(pred[i]) for i in k)
                row.update(model_score=s,
                           model_kind="flip" if s > DEC_T else "noflip")
            rows.append(row)
    return pd.DataFrame(rows)


def test3(model, batch_size, seed=0):
    act, info = episodes_info("parcel-sort-b")
    ds = open_ds(model, "parcel-sort-b")
    rng = np.random.default_rng(seed)
    point = {e: ep["start"] + max(ep["onset"] - DEC_GAP, ep["grasp"])
             for e, ep in info.items()}
    pairs = []
    for e, ep in info.items():
        other = [o for o in info if info[o]["kind"] != ep["kind"]]
        pairs.append((e, int(rng.choice(other))))
    items = []
    for e, o in pairs:
        it = dict(ds[point[e]])
        donor = ds[point[o]]
        for k in donor:
            if k.startswith("observation.images"):
                it[k] = donor[k]
        items.append(it)
    pred = predict_rows(model, ds, list(range(len(items))), batch_size,
                        items=items)
    rows = []
    for (e, o), p in zip(pairs, pred):
        kind = "flip" if roll_score(p) > DEC_T else "noflip"
        rows.append((e, o, info[e]["kind"], info[o]["kind"], kind))
    return pd.DataFrame(rows, columns=["episode", "donor", "state_kind",
                                       "image_kind", "pred_kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "offline_eval_b"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    probe = pd.concat([test2(None, TRAIN_DEC, "train", 0),
                       test2(None, HELD_OUT, "held_out", 0)])
    ok = probe.probe_kind == probe.truth
    print("demo probe:", probe.groupby("set").apply(
        lambda d: f"{(d.probe_kind == d.truth).sum()}/{len(d)}").to_dict(),
        "| kinds:", probe.groupby(["set", "truth"]).size().to_dict())
    print("probe misses:", probe[~ok][["set", "dataset", "episode",
                                       "truth"]].values.tolist())

    t1, t2, t3 = [], [], []
    for name in args.models:
        run, step = SPECS[name]
        oe.MODELS[name] = TRAIN / run
        model = oe.Model(name, step)
        print(f"== {name} ({model.path})", flush=True)
        for parts, set_name in ((HELD_OUT, "held_out"),
                                (TRAIN_REF, "train_ref")):
            if not parts:
                continue
            df = test1(model, parts, set_name, args.every, args.batch)
            df["model"] = name
            t1.append(df)
            print(f"  test 1 {set_name}: {len(df)} samples, arm MAE "
                  f"{df.arm_mae.mean():.2f} deg", flush=True)
        for parts, set_name in ((TRAIN_DEC, "train"),
                                (HELD_OUT, "held_out")):
            if not parts:
                continue
            df = test2(model, parts, set_name, args.batch)
            df["model"] = name
            t2.append(df)
            print(f"  test 2 {set_name}: "
                  f"{(df.model_kind == df.truth).sum()}/{len(df)} correct",
                  flush=True)
        df = test3(model, args.batch)
        df["model"] = name
        t3.append(df)
        print(f"  test 3: follows images "
              f"{(df.pred_kind == df.image_kind).mean():.2f}, follows "
              f"arm state {(df.pred_kind == df.state_kind).mean():.2f}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    t1 = pd.concat(t1, ignore_index=True)
    t2 = pd.concat(t2, ignore_index=True)
    t3 = pd.concat(t3, ignore_index=True)
    t1.to_csv(out / "test1_samples.csv", index=False)
    t2.to_csv(out / "test2_decisions.csv", index=False)
    t3.to_csv(out / "test3_image_swap.csv", index=False)
    s1 = t1.pivot_table(index=["model", "set"], columns="phase",
                        values="arm_mae", aggfunc="mean")
    s1["all"] = t1.groupby(["model", "set"]).arm_mae.mean()
    s1["gripper"] = t1.groupby(["model", "set"]).grip_mae.mean()
    print("\nTest 1: next 1 s action error (deg, arm joints)")
    print(s1[["approach", "carry", "exit", "all", "gripper"]]
          .round(2).to_string())
    t2["ok"] = t2.model_kind == t2.truth
    s2 = t2.groupby(["model", "set", "truth"]).ok.agg(["sum", "size"])
    print("\nTest 2: flip decision (correct / n)")
    print(s2.to_string())
    t3["follows_images"] = t3.pred_kind == t3.image_kind
    s3 = t3.groupby(["model", "state_kind"]).follows_images.mean()
    print("\nTest 3: image swap, follows the images")
    print(s3.round(2).to_string())
    s1.round(3).to_csv(out / "test1_action_error.csv")
    s2.to_csv(out / "test2_summary.csv")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
