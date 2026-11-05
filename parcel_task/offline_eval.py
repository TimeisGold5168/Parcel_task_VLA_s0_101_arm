"""Offline evaluation of the parcel models (no robot).

Test 1 (action error): from sampled frames, predict the next 1 s
(30 frames) of actions and compare with the human demo. Mean absolute
error in degrees, 5 arm joints and the gripper apart, per phase
(push | flip | exit, from phase_marks.csv).

Test 2 (label decision): at the end of the push (mark 1) the model must
choose "flip" (label up) or "push to exit" (label down). From mark 1 +
0.5 s and + 1 s, the predicted next 1 s, relative to the current
action, is compared with the futures of all A1 training episodes at the
same point. The decision is the kind (flip or no flip) with the smaller
mean distance. The same probe on the demo futures shows how well the
probe itself separates the two kinds.

Held-out episodes (not in training): parcel-sort-a_backup_with_h2
episodes 7-9 (first heap 2 attempt) and parcel-sort-a_heap1_720
episodes 3-6 (top camera 1280x720, resized to 848x480; episodes 0-2 have
dropped top frames). Reference: 7 training episodes of parcel-sort-a.

Usage:
  python outputs/portfolio/parcel_task/offline_eval.py
  python outputs/portfolio/parcel_task/offline_eval.py --models smolvla
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bsp"))
from train_mixed import patch_diffusion  # noqa: E402
from bspline import make_spline, segment_range  # noqa: E402
from segment import predict_segment  # noqa: E402

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
TRAIN = HERE.parent.parent / "train"
MODELS = {
    "diffusion": TRAIN / "diffusion_parcel_a1_80k",
    "diffusion_bsp": TRAIN / "diffusion_bsp_parcel_40k",
    "smolvla": TRAIN / "smolvla_parcel_a1_20k",
}
# set name: [(dataset, episodes, decision frames only)]
SETS = {
    "held_out": [("parcel-sort-a_backup_with_h2", [7, 8, 9], False),
                 ("parcel-sort-a_heap1_720", [3, 4, 5, 6], False)],
    "train_ref": [("parcel-sort-a", [3, 17, 31, 45, 59, 73, 87], False)],
    "train_dec": [("parcel-sort-a", list(range(100)), True)],
}
TOP = "observation.images.top"
H = 30  # 1 s at 30 fps
# the flip / exit choice shows in the demo motion from ~0.5 s after
# mark 1 (probe on A1 demos: 62/100 at +0, 89/100 at +15, 92/100 at +30)
DEC_OFFSETS = (15, 30)


def load_actions(root):
    files = sorted((root / "data").rglob("*.parquet"))
    d = pd.concat(pd.read_parquet(f, columns=["index", "episode_index",
                                              "frame_index", "action"])
                  for f in files).sort_values("index")
    return d.reset_index(drop=True)


def load_marks(root):
    m = pd.read_csv(root / "phase_marks.csv")
    return {e: sorted(g.frame.tolist()) for e, g in m.groupby("episode")}


def phase_of(frame, marks):
    if not marks:
        return "push"
    if frame < marks[0]:
        return "push"
    if len(marks) == 2 and frame < marks[1]:
        return "flip"
    return "exit"


class Model:
    """One checkpoint with its processors and a 30-frame predictor."""

    def __init__(self, name, step):
        ck = MODELS[name] / "checkpoints" / step / "pretrained_model"
        self.name, self.path = name, ck
        self.bsp = (json.loads((ck / "bsp.json").read_text())
                    if (ck / "bsp.json").exists() else None)
        cfg = PreTrainedConfig.from_pretrained(ck)
        if cfg.type == "diffusion":
            patch_diffusion()
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(ck)
        self.policy.to("cuda").eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(ck))

    def delta(self, fps):
        idx = self.cfg.observation_delta_indices
        return None if idx is None else [i / fps for i in idx]

    @torch.no_grad()
    def predict(self, batch):
        """(B, 30, 6) actions in degrees."""
        obs = self.pre({k: v for k, v in batch.items()
                        if not k.startswith("action")})
        p = self.policy
        if self.cfg.type == "smolvla":
            p.reset()
            out = self.post(p.predict_action_chunk(obs))
            return out[:, :H].float().cpu().numpy()
        seg = self.post(predict_segment(p, obs)).float().cpu().numpy()
        if self.bsp is None:  # plain diffusion: rows from n_obs_steps-1
            start = self.cfg.n_obs_steps - 1
            return seg[:, start:start + H]
        # knots are relative to the current frame; frames outside the
        # valid segment range stay NaN (not scored)
        deg = self.bsp["degree"]
        out = np.full((len(seg), H, seg.shape[2] - 1), np.nan, np.float32)
        t = np.arange(H, dtype=np.float64)
        for b, params in enumerate(seg):
            t0, t1 = segment_range(params, deg)
            ok = (t >= t0) & (t < t1)
            if ok.any():
                out[b, ok] = make_spline(params, deg)(t[ok])
        return out


def frames_for(acts, episodes, marks, every, dec_only=False):
    """Sample frames with 30 future frames inside the episode.

    Also adds the decision frames mark 1 + DEC_OFFSETS (phase "dec+N").
    """
    rows = []
    for e in episodes:
        ep = acts[acts.episode_index == e]
        n = len(ep)
        if not dec_only:
            for f in range(0, n - H, every):
                rows.append((int(ep["index"].iloc[f]), e, f,
                             phase_of(f, marks.get(e, []))))
        m = marks.get(e, [])
        for off in DEC_OFFSETS:
            f = m[0] + off if m else n
            if f + H < n:
                rows.append((int(ep["index"].iloc[f]), e, f, f"dec+{off}"))
    return rows


def run_set(model, ds_name, episodes, every, batch_size, dec_only=False):
    root = LOW / ds_name
    acts = load_actions(root)
    marks = load_marks(root)
    act = np.stack(acts.action.values)
    meta_fps = 30
    d = model.delta(meta_fps)
    delta = None
    if d is not None:
        info = json.loads((root / "meta/info.json").read_text())
        delta = {k: d for k in info["features"]
                 if k.startswith("observation")}
    ds = LeRobotDataset("Low101/" + ds_name, root=root,
                        delta_timestamps=delta)
    rows = frames_for(acts, episodes, marks, every, dec_only)
    preds = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        items = [ds[r[0]] for r in chunk]
        batch = torch.utils.data.default_collate(items)
        img = batch[TOP]
        if img.shape[-2:] != (480, 848):  # 720p heap 1: same 16:9 view
            lead = img.shape[:-3]
            img = F.interpolate(img.reshape(-1, *img.shape[-3:]),
                                size=(480, 848), mode="bilinear",
                                antialias=True, align_corners=False)
            batch[TOP] = img.reshape(*lead, *img.shape[-3:])
        torch.manual_seed(i)
        preds.append(model.predict(batch))
    pred = np.concatenate(preds)
    gt = np.stack([act[r[0]:r[0] + H] for r in rows])
    now = np.stack([act[r[0]] for r in rows])
    info = pd.DataFrame(rows, columns=["index", "episode", "frame",
                                       "phase"])
    info["dataset"] = ds_name
    return info, pred, gt, now


def reference_futures(off):
    """A1 futures from mark 1 + off, relative to the action there."""
    root = LOW / "parcel-sort-a"
    acts = load_actions(root)
    act = np.stack(acts.action.values)
    marks = load_marks(root)
    refs = {"flip": [], "noflip": []}
    for e, m in marks.items():
        g = acts.index[acts.episode_index == e]
        i = int(g[0]) + m[0] + off
        if m[0] + off + H < len(g):
            kind = "flip" if len(m) == 2 else "noflip"
            refs[kind].append((e, act[i:i + H] - act[i]))
    return refs


def decide(future_rel, refs, skip=()):
    dist = {}
    for kind, lst in refs.items():
        dist[kind] = np.mean([np.nanmean(np.abs(future_rel - r))
                              for e, r in lst if e not in skip])
    return min(dist, key=dist.get), dist


def swap_test(model, refs, off, batch_size, seed=0):
    """Test 3: keep the arm state, swap in all camera images of an A1
    episode with the other label (same point after mark 1)."""
    root = LOW / "parcel-sort-a"
    acts = load_actions(root)
    act = np.stack(acts.action.values)
    marks = load_marks(root)
    d = model.delta(30)
    info = json.loads((root / "meta/info.json").read_text())
    delta = (None if d is None else
             {k: d for k in info["features"] if k.startswith("observation")})
    ds = LeRobotDataset("Low101/parcel-sort-a", root=root,
                        delta_timestamps=delta)
    start = {e: int(acts.index[acts.episode_index == e][0]) for e in marks}
    kind = {e: "flip" if len(m) == 2 else "noflip" for e, m in marks.items()}
    rng = np.random.default_rng(seed)
    pairs = []
    for e, m in marks.items():
        other = [o for o in marks if kind[o] != kind[e]]
        pairs.append((e, int(rng.choice(other))))
    rows = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        items = []
        for e, o in chunk:
            it = dict(ds[start[e] + marks[e][0] + off])
            donor = ds[start[o] + marks[o][0] + off]
            for k in donor:
                if k.startswith("observation.images"):
                    it[k] = donor[k]
            items.append(it)
        torch.manual_seed(i)
        pred = model.predict(torch.utils.data.default_collate(items))
        for (e, o), p in zip(chunk, pred):
            now = act[start[e] + marks[e][0] + off]
            k, _ = decide(p - now, refs, skip=(e, o))
            rows.append((e, o, kind[e], kind[o], k))
    return pd.DataFrame(rows, columns=["episode", "donor", "state_kind",
                                       "image_kind", "pred_kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--step", default="last")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "offline_eval"))
    ap.add_argument("--swap-only", action="store_true",
                    help="run only Test 3 (image swap after mark 1)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = {off: reference_futures(off) for off in DEC_OFFSETS}
    print(f"reference futures: {len(refs[15]['flip'])} flip, "
          f"{len(refs[15]['noflip'])} no flip")
    if args.swap_only:
        res = []
        for name in args.models:
            model = Model(name, args.step)
            for off in DEC_OFFSETS:
                r = swap_test(model, refs[off], off, args.batch)
                r["model"], r["phase"] = name, f"dec+{off}"
                res.append(r)
                follow = (r.pred_kind == r.image_kind).mean()
                print(f"{name} dec+{off}: follows swapped images "
                      f"{follow:.2f}, follows arm state "
                      f"{(r.pred_kind == r.state_kind).mean():.2f}",
                      flush=True)
            del model
            torch.cuda.empty_cache()
        res = pd.concat(res, ignore_index=True)
        res.to_csv(out / "test3_image_swap.csv", index=False)
        return

    rows = []
    for name in args.models:
        model = Model(name, args.step)
        print(f"== {name} ({model.path})", flush=True)
        for set_name, parts in SETS.items():
            for ds_name, eps, dec_only in parts:
                info, pred, gt, now = run_set(model, ds_name, eps,
                                              args.every, args.batch,
                                              dec_only)
                err = np.abs(pred - gt)
                info["arm_mae"] = np.nanmean(err[:, :, :5], axis=(1, 2))
                info["grip_mae"] = np.nanmean(err[:, :, 5], axis=1)
                info["scored"] = (~np.isnan(err[:, :, 0])).sum(axis=1)
                info["model"], info["set"] = name, set_name
                dec = []
                for k, r in info.iterrows():
                    if not r.phase.startswith("dec"):
                        dec.append(("", "", ""))
                        continue
                    ref = refs[int(r.phase[4:])]
                    marks = load_marks(LOW / ds_name)[r.episode]
                    truth = "flip" if len(marks) == 2 else "noflip"
                    skip = ((r.episode,) if ds_name == "parcel-sort-a"
                            else ())
                    p_kind, _ = decide(pred[k] - now[k], ref, skip)
                    g_kind, _ = decide(gt[k] - now[k], ref, skip)
                    dec.append((truth, p_kind, g_kind))
                info[["truth", "pred_kind", "demo_kind"]] = dec
                rows.append(info)
                print(f"  {set_name} {ds_name}: {len(info)} samples, "
                      f"arm MAE {info.arm_mae.mean():.2f} deg", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "samples.csv", index=False)
    s1 = df[~df.phase.str.startswith("dec")]
    t1 = s1.pivot_table(index=["model", "set"], columns="phase",
                        values="arm_mae", aggfunc="mean")
    t1["all"] = s1.groupby(["model", "set"]).arm_mae.mean()
    t1["gripper"] = s1.groupby(["model", "set"]).grip_mae.mean()
    t1["frames"] = s1.groupby(["model", "set"]).scored.mean()
    print("\nTest 1: next 1 s action error (deg, arm joints)")
    print(t1[["push", "flip", "exit", "all", "gripper", "frames"]]
          .round(2).to_string())
    m1 = df[df.phase.str.startswith("dec")].copy()
    m1["model_ok"] = m1.pred_kind == m1.truth
    m1["probe_ok"] = m1.demo_kind == m1.truth
    t2 = m1.groupby(["model", "set", "phase"]).agg(
        n=("model_ok", "size"), model_correct=("model_ok", "sum"),
        probe_correct=("probe_ok", "sum"))
    print("\nTest 2: label decision after mark 1 (correct / n)")
    print(t2.to_string())
    print("\nheld-out decisions:")
    print(m1[m1.set == "held_out"][["model", "dataset", "episode",
                                    "phase", "truth", "pred_kind",
                                    "demo_kind"]].to_string(index=False))
    t1.round(3).to_csv(out / "test1_action_error.csv")
    t2.to_csv(out / "test2_label_decision.csv")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
