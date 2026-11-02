"""Train ACT-BSP, Diffusion-BSP or SmolVLA-BSP (warm start).

The policy predicts one B-spline segment per frame: a (rows, 1 + 6)
matrix (rows = chunk_size + 2 * degree; column 0 = knots in frames
relative to now, columns 1-6 = joint control points). Targets come from
make_targets.py. Weights with matching shapes are copied from the
baseline checkpoint; the action input/output layers start fresh.
SmolVLA pads actions to max_action_dim, so every weight is copied.

Usage:
  python outputs/portfolio/bsp/train_bsp.py --policy act \
    --init outputs/train/act_all_ft50k/checkpoints/050000/pretrained_model \
    --out outputs/train/act_bsp_ws30k --steps 30000 --batch 8
  add --smoke for a 20-step test that also checks decoding.

Parcel task (3 cameras, targets_parcel-sort-a_e0.5.npz):
  python outputs/portfolio/bsp/train_bsp.py --policy smolvla \
    --dataset parcel-sort-a --init <smolvla checkpoint or hub id> \
    --out outputs/train/smolvla_bsp_parcel --steps 20000 --batch 12
  SmolVLA uses the processors of --init (task tokenizer, camera
  rename top/wrist/side -> camera1/2/3, see --rename). Diffusion with
  mixed camera sizes uses the per-camera resize of train_mixed.py.
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
from lerobot.utils.feature_utils import dataset_to_policy_features

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "parcel_task"))
from bspline import decode  # noqa: E402
from segment import predict_segment  # noqa: E402
from train_mixed import patch_diffusion  # noqa: E402

ROOT = Path.home() / ".cache/huggingface/lerobot/Low101"
PARCEL_RENAME = {"observation.images.top": "observation.images.camera1",
                 "observation.images.wrist": "observation.images.camera2",
                 "observation.images.side": "observation.images.camera3"}


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


def build_config(args, rows, dim, meta=None):
    cfg = PreTrainedConfig.from_pretrained(args.init)
    cfg.pretrained_path = None
    cfg.device = args.device
    if cfg.type == "act":
        cfg.chunk_size = rows
        cfg.n_action_steps = rows
        cfg.temporal_ensemble_coeff = None
    elif cfg.type == "diffusion":
        cfg.horizon = rows
        cfg.n_action_steps = rows
        cfg.drop_n_last_frames = 0
    elif cfg.type == "smolvla":
        assert dim <= cfg.max_action_dim, "segment wider than max_action_dim"
        cfg.chunk_size = rows
        cfg.n_action_steps = rows
        cfg.load_vlm_weights = False  # all weights come from --init
        # inputs from the dataset, renamed like the processors of --init
        feats = dataset_to_policy_features(meta.features)
        cfg.input_features = {
            args.rename.get(k, k): f for k, f in feats.items()
            if f.type is not FeatureType.ACTION}
        n_cam = sum(f.type is FeatureType.VISUAL
                    for f in cfg.input_features.values())
        if n_cam >= 3:  # all 3 pretrained camera slots are real
            cfg.empty_cameras = 0
    else:
        raise ValueError(f"unsupported policy type {cfg.type}")
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}
    return cfg


def make_processors(cfg, args, stats):
    """ACT/Diffusion: new processors. SmolVLA: the processors of --init
    (tokenizer, camera rename) with the B-spline action stats."""
    if cfg.type != "smolvla":
        return make_pre_post_processors(cfg, dataset_stats=stats)
    norm = {"features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping, "stats": stats}
    return make_pre_post_processors(
        cfg, pretrained_path=str(args.init),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "normalizer_processor": norm,
            "rename_observations_processor": {"rename_map": args.rename}},
        postprocessor_overrides={"unnormalizer_processor": {
            "features": cfg.output_features,
            "norm_map": cfg.normalization_mapping, "stats": stats}})


def resolve_init(init):
    """Local checkpoint folder, or a Hub model id (downloaded)."""
    if Path(init).exists():
        return Path(init)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(init))


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
        if "task" in batch:  # SmolVLA tokenizes the task text
            obs["task"] = batch["task"][:1]
        obs = pre(obs)
        chunk = post(predict_segment(policy, obs))[0].cpu().numpy()
    policy.train()
    acts = decode(chunk, 8, degree)
    print("predicted knots:", np.round(chunk[:, 0], 1))
    print("decoded 8 actions, first:", np.round(acts[0], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["act", "diffusion", "smolvla"],
                    required=True)
    ap.add_argument("--init", required=True,
                    help="checkpoint folder or Hub model id")
    ap.add_argument("--dataset", default="third-act-attempt")
    ap.add_argument("--targets", default=None,
                    help="default: bsp/targets_<dataset>_e0.5.npz")
    ap.add_argument("--rename", default=None,
                    help="SmolVLA camera rename map (JSON); default: "
                    "top/wrist/side -> camera1/2/3")
    ap.add_argument("--device", default="cuda")
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
    if args.targets is None:
        args.targets = str(Path(__file__).parent
                           / f"targets_{args.dataset}_e0.5.npz")
    args.rename = (json.loads(args.rename) if args.rename
                   else PARCEL_RENAME)
    args.init = resolve_init(args.init)

    tz = np.load(args.targets)
    targets = tz["targets"]
    degree = int(tz["degree"])
    rows, dim = targets.shape[1:]
    meta_bsp = {"degree": degree, "chunk_size": int(tz["chunk_size"]),
                "max_error": float(tz["max_error"]), "fps": int(tz["fps"]),
                "rows": int(rows), "targets": Path(args.targets).name}

    meta = LeRobotDatasetMetadata("Low101/" + args.dataset,
                                  root=ROOT / args.dataset)
    cfg = build_config(args, rows, dim, meta)
    assert cfg.type == args.policy, f"--init is a {cfg.type} checkpoint"
    if cfg.type == "diffusion":
        patch_diffusion()  # no change when all cameras have one size
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
    pre, post = make_processors(cfg, args, stats)
    # parcel Diffusion (3 cameras) needs AMP to fit in 6 GB
    amp = bool(getattr(cfg, "use_amp", False)) and args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

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
          f"steps {args.steps}, batch {args.batch}, amp {amp}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bsp_train_args.json").write_text(
        json.dumps({**vars(args), **meta_bsp}, indent=1, default=str))

    policy.train()
    step, t0, run_loss = 0, time.time(), 0.0
    batch = None
    while step < args.steps:
        for batch in loader:
            batch = pre(batch)
            with torch.autocast("cuda", enabled=amp):
                loss, _ = policy.forward(batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
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
