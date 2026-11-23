"""Offline grid test of the BSP executor settings on dataset episodes.

Plans come from the real Diffusion-BSP policy on demo frames; the
executor (replan timing, start search, blend, max-step clamp) is
simulated at --hz. The observation at episode time tau is the demo
frame at tau, so this measures how smooth and faithful the command
stream is, not task success (no closed loop).

Grid: DDIM steps x noise (random / fixed per episode) x speed x
max-step x blend. Plan latency = measured denoise time + overhead.

Output: runs/grid_executor.csv and summary tables.

Usage:
  python -m parcel_vla.bsp.grid_executor --episodes 4
"""

import argparse
import csv
import glob
import itertools
import json

import numpy as np
import pandas as pd
import torch

from parcel_vla import paths
from parcel_vla.bsp.bspline import make_spline, segment_range
from parcel_vla.robot.run_bsp import RUNS

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

ROOT = paths.LOW
# denoise time (ms, batch 1, measured 2026-10-25) + robot-side overhead
DENOISE_MS = {10: 88, 5: 48, 3: 32}
OVERHEAD_MS = 40
ARM = slice(0, 5)


class Planner:
    def __init__(self, dataset):
        path = RUNS["diffusion"]
        self.meta = json.loads((path / "bsp.json").read_text())
        self.cfg = PreTrainedConfig.from_pretrained(path)
        pol = get_policy_class(self.cfg.type).from_pretrained(path)
        self.model = pol.to("cuda").eval().diffusion
        self.pre, self.post = make_pre_post_processors(
            self.cfg, pretrained_path=str(path))
        self.ds = LeRobotDataset(paths.dataset_id(dataset), root=ROOT / dataset)

    @torch.inference_mode()
    def plans(self, idx, steps, noise, batch=16):
        """Segments for dataset frames idx; noise None = random."""
        cfg, model, rows = self.cfg, self.model, self.meta["rows"]
        model.num_inference_steps = steps
        out = []
        for k in range(0, len(idx), batch):
            items = [self.ds[int(i)] for i in idx[k:k + batch]]
            obs = {key: torch.stack([it[key] for it in items]).cuda()
                   for key in items[0] if key.startswith("observation")}
            b = dict(self.pre(obs))
            n = cfg.n_obs_steps
            for key in cfg.image_features:
                b[key] = b[key].unsqueeze(1).expand(-1, n, -1, -1, -1)
            b[OBS_STATE] = b[OBS_STATE].unsqueeze(1).expand(-1, n, -1)
            b[OBS_IMAGES] = torch.stack(
                [b[key] for key in cfg.image_features], dim=-4)
            nz = (None if noise is None
                  else noise.expand(len(items), rows, -1).contiguous())
            seg = model.conditional_sample(
                len(items), global_cond=model._prepare_global_conditioning(b),
                noise=nz)
            out.append(self.post(seg).float().cpu().numpy())
        return np.concatenate(out)


def simulate(plans, demo, fps, degree, speed, max_step, blend, latency,
             hz=50.0, replan_frac=0.5):
    """Run the executor logic on precomputed plans (every 2nd frame)."""
    n_frames = len(demo)
    dt = 1.0 / hz
    seg, old, last = None, None, None
    pending = None                       # (ready_wall, tau_obs, params)
    blend_until = blend_from = 0.0
    log, seams, holds, clamps = [], [], 0, 0
    w = 0.0
    while True:
        tau = w * fps * speed
        if tau >= n_frames - 1:
            break
        # planner
        if pending is not None and w >= pending[0]:
            _, tau_obs, p = pending
            pending = None
            spl = make_spline(p, degree)
            t0, t1 = segment_range(p, degree)
            natural = tau - tau_obs
            if last is not None:
                grid = np.clip(np.arange(natural - 3, natural + 6, 0.25),
                               t0, t1 - 1e-6)
                err = np.abs(spl(grid) - last).sum(1)
                start = float(grid[int(np.nanargmin(err))])
                seams.append(np.abs(spl(start) - last)[ARM].max())
            else:
                start = float(np.clip(natural, t0, t1 - 1e-6))
            old = seg
            seg = (spl, start, tau, t0, t1)
            if old is not None and blend > 0:
                o_spl, o_start, o_now, _, o_t1 = old
                left = (o_t1 - (o_start + tau - o_now)) / (fps * speed)
                blend_from, blend_until = w, w + min(blend, max(left, 0))
        if pending is None:
            prog = 1.0
            if seg is not None:
                spl, start, now0, t0, t1 = seg
                prog = (start + tau - now0 - t0) / max(t1 - t0, 1e-6)
            if prog >= replan_frac:
                f = min(int(round(tau / 2)) * 2, (len(plans) - 1) * 2)
                pending = (w + latency, tau, plans[f // 2])
        # control
        if seg is not None:
            spl, start, now0, t0, t1 = seg
            s_raw = start + tau - now0
            holds += s_raw >= t1 - 1e-3
            target = spl(np.clip(s_raw, t0, t1 - 1e-6))
            if old is not None and w < blend_until:
                o_spl, o_start, o_now, o_t0, o_t1 = old
                o_s = np.clip(o_start + tau - o_now, o_t0, o_t1 - 1e-6)
                a = (w - blend_from) / max(blend_until - blend_from, 1e-6)
                a = a * a * (3 - 2 * a)           # smoothstep
                target = (1 - a) * o_spl(o_s) + a * target
            if last is not None:
                step = target - last
                clamps += bool((np.abs(step[ARM]) > max_step).any())
                target = last + np.clip(step, -max_step, max_step)
            last = target
            log.append((w, tau, *target))
        w += dt
    log = np.array(log)
    if len(log) < 20:
        return None
    cmd, taus = log[:, 2:], log[:, 1]
    ref = np.stack([np.interp(taus, np.arange(n_frames), demo[:, j])
                    for j in range(demo.shape[1])], 1)
    err = np.abs(cmd - ref)[:, ARM].max(1)
    vel = np.diff(cmd[:, ARM], axis=0) * hz
    moving = np.abs(vel) > 5.0
    flips = ((np.sign(vel[1:]) != np.sign(vel[:-1]))
             & moving[1:] & moving[:-1]).sum()
    jerk = np.abs(np.diff(cmd[:, ARM], n=3, axis=0)) * hz ** 3
    dur = log[-1, 0]
    return {"err_med": float(np.median(err)),
            "err_p95": float(np.percentile(err, 95)),
            "seam_p95": float(np.percentile(seams, 95)) if seams else 0.0,
            "clamp_pct": 100 * clamps / len(log),
            "hold_pct": 100 * holds / len(log),
            "rev_per_s": float(flips / dur),
            "jerk_p95_k": float(np.percentile(jerk.max(1), 95) / 1000),
            "dur_s": float(dur)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="third-act-attempt")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--speeds", default="1,1.5,2,2.5,3")
    ap.add_argument("--max-steps", default="4,5,6,8")
    ap.add_argument("--blends", default="0,0.1,0.2,0.3")
    ap.add_argument("--ddim", default="10,5,3")
    args = ap.parse_args()
    speeds = [float(x) for x in args.speeds.split(",")]
    max_steps = [float(x) for x in args.max_steps.split(",")]
    blends = [float(x) for x in args.blends.split(",")]
    ddims = [int(x) for x in args.ddim.split(",")]

    pl = Planner(args.dataset)
    fps, degree = pl.meta["fps"], pl.meta["degree"]
    rows = pl.meta["rows"]
    root = ROOT / args.dataset
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(
        str(root / "data/**/*.parquet"), recursive=True)).sort_values("index")
    eps = np.random.default_rng(0).choice(
        df["episode_index"].unique(), args.episodes, replace=False)

    results = []
    for e in eps:
        d = df[df["episode_index"] == e]
        idx = d["index"].values
        demo = np.stack(d["action"].values)
        g = torch.Generator("cuda").manual_seed(int(e))
        fixed = torch.randn((1, rows, 7), device="cuda", generator=g)
        for steps in ddims:
            for mode in ("random", "fixed"):
                plans = pl.plans(idx[::2], steps,
                                 fixed if mode == "fixed" else None)
                lat = (DENOISE_MS[steps] + OVERHEAD_MS) / 1000
                for sp, ms, bl in itertools.product(speeds, max_steps,
                                                    blends):
                    r = simulate(plans, demo, fps, degree, sp, ms, bl, lat)
                    if r:
                        results.append({"episode": int(e), "ddim": steps,
                                        "noise": mode, "speed": sp,
                                        "max_step": ms, "blend": bl, **r})
        print(f"episode {e} done ({len(d)} frames)", flush=True)

    out = paths.RUNS / "grid_executor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"saved {out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
