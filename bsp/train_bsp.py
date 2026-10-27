"""Train ACT-BSP or Diffusion-BSP on a LeRobot dataset (warm start).

The policy predicts one B-spline segment per frame: a (rows, 1 + 6)
matrix (rows = chunk_size + 2 * degree; column 0 = knots in frames
relative to now, columns 1-6 = joint control points). Targets come from
make_targets.py. Weights with matching shapes are copied from the
baseline checkpoint; the action input/output layers start fresh.

Usage:
  python outputs/portfolio/bsp/train_bsp.py --policy act \
    --init outputs/train/act_all_ft50k/checkpoints/050000/pretrained_model \
    --out outputs/train/act_bsp_ws30k --steps 30000 --batch 8
  add --smoke for a 20-step test that also checks decoding.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

sys.path.insert(0, str(Path(__file__).parent))
from bspline import decode  # noqa: E402
from segment import predict_segment  # noqa: E402

ROOT = Path.home() / ".cache/huggingface/lerobot/Low101"


class BSplineTargets(torch.utils.data.Dataset):
    """Dataset items with the action replaced by the B-spline segment."""

    def __init__(self, ds, targets):
        self.ds, self.targets = ds, targets
        self.rows = targets.shape[1]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        g = int(item["index"])
        item["action"] = torch.from_numpy(self.targets[g])
        item["action_is_pad"] = torch.zeros(self.rows, dtype=torch.bool)
        return item


def action_stats(targets):
    t = targets.astype(np.float64)
    return {"mean": t.mean(0), "std": t.std(0), "min": t.min(0),
            "max": t.max(0), "count": np.array([len(t)])}


def build_config(args, rows, dim):
    cfg = PreTrainedConfig.from_pretrained(args.init)
    cfg.pretrained_path = None
    cfg.device = "cuda"
    if cfg.type == "act":
        cfg.chunk_size = rows
        cfg.n_action_steps = rows
        cfg.temporal_ensemble_coeff = None
    elif cfg.type == "diffusion":
        cfg.horizon = rows
        cfg.n_action_steps = rows
        cfg.drop_n_last_frames = 0
    else:
        raise ValueError(f"unsupported policy type {cfg.type}")
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}
    return cfg


def warm_start(policy, init_dir, rows):
    """Copy weights with matching shapes; slice ACT query embeddings."""
    src = load_file(str(Path(init_dir) / "model.safetensors"))
    own = policy.state_dict()
    loaded, sliced, skipped = 0, [], []
    for k, v in src.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
            loaded += 1
        elif (v.dim() == own[k].dim() and v.shape[1:] == own[k].shape[1:]
              and v.shape[0] >= own[k].shape[0]):
            own[k] = v[: own[k].shape[0]]
            sliced.append(k)
        else:
            skipped.append(f"{k} {tuple(v.shape)}->{tuple(own[k].shape)}")
    policy.load_state_dict(own)
    print(f"warm start: {loaded} tensors copied, sliced {sliced}")
    print("fresh (shape changed):", *skipped, sep="\n  ")


def save(policy, pre, post, out, step, meta):
    d = Path(out) / "checkpoints" / f"{step:06d}" / "pretrained_model"
    d.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(d)
    pre.save_pretrained(d)
    post.save_pretrained(d)
    (d / "bsp.json").write_text(json.dumps(meta, indent=1))
    last = Path(out) / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(f"{step:06d}")
    print(f"saved {d}")


def check_decode(policy, pre, post, batch, degree):
    """Predict one segment and decode it into joint actions."""
    policy.eval()
    with torch.no_grad():
        obs = {k: v[:1] for k, v in batch.items()
               if k.startswith("observation")}
        obs = pre(obs)
        chunk = post(predict_segment(policy, obs))[0].cpu().numpy()
    policy.train()
    acts = decode(chunk, 8, degree)
    print("predicted knots:", np.round(chunk[:, 0], 1))
    print("decoded 8 actions, first:", np.round(acts[0], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["act", "diffusion"], required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--dataset", default="third-act-attempt")
    ap.add_argument("--targets", default=str(
        Path(__file__).parent / "targets_third-act-attempt_e0.5.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.smoke:
        args.steps, args.save_freq, args.log_freq = 20, 20, 5

    tz = np.load(args.targets)
    targets = tz["targets"]
    degree = int(tz["degree"])
    rows, dim = targets.shape[1:]
    meta_bsp = {"degree": degree, "chunk_size": int(tz["chunk_size"]),
                "max_error": float(tz["max_error"]), "fps": int(tz["fps"]),
                "rows": int(rows), "targets": Path(args.targets).name}

    cfg = build_config(args, rows, dim)
    assert cfg.type == args.policy, f"--init is a {cfg.type} checkpoint"
    meta = LeRobotDatasetMetadata("Low101/" + args.dataset,
                                  root=ROOT / args.dataset)
    fps = meta.fps
    obs_idx = cfg.observation_delta_indices
    delta = None
    if obs_idx is not None:
        delta = {k: [i / fps for i in obs_idx] for k in meta.features
                 if k.startswith("observation")}
    ds = LeRobotDataset("Low101/" + args.dataset, root=ROOT / args.dataset,
                        delta_timestamps=delta)
    assert len(targets) == ds.meta.total_frames, "targets/dataset mismatch"
    data = BSplineTargets(ds, targets)

    stats = {k: dict(v) for k, v in meta.stats.items()}
    stats["action"] = action_stats(targets)
    policy = get_policy_class(cfg.type)(config=cfg)
    warm_start(policy, args.init, rows)
    policy.to(cfg.device)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)

    opt_cfg = cfg.get_optimizer_preset()
    optimizer = opt_cfg.build(policy.get_optim_params())
    sched_cfg = cfg.get_scheduler_preset()
    scheduler = (sched_cfg.build(optimizer, args.steps)
                 if sched_cfg is not None else None)
    clip = getattr(opt_cfg, "grad_clip_norm", 10.0)

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    print(f"{cfg.type}-BSP: {len(data)} frames, rows {rows}, dim {dim}, "
          f"steps {args.steps}, batch {args.batch}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bsp_train_args.json").write_text(
        json.dumps({**vars(args), **meta_bsp}, indent=1))

    policy.train()
    step, t0, run_loss = 0, time.time(), 0.0
    batch = None
    while step < args.steps:
        for batch in loader:
            batch = pre(batch)
            loss, _ = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            run_loss += loss.item()
            if step % args.log_freq == 0:
                rate = step / (time.time() - t0)
                eta = (args.steps - step) / rate / 3600
                print(f"step {step} loss {run_loss / args.log_freq:.4f} "
                      f"{rate:.2f} step/s eta {eta:.1f} h", flush=True)
                run_loss = 0.0
            if step % args.save_freq == 0 or step == args.steps:
                save(policy, pre, post, args.out, step, meta_bsp)
            if step >= args.steps:
                break
    if args.smoke:
        raw = next(iter(loader))
        check_decode(policy, pre, post, raw, degree)


if __name__ == "__main__":
    main()
