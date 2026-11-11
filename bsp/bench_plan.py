"""Benchmark and correctness check for one SmolVLA(-BSP) plan.

Fixed inputs (one A1 training frame per sample, fixed noise), warm-up,
CUDA sync, many repeats. Prints p50 / p95 plan time and the output
difference against a saved eager reference.

  # 1. save the eager reference (and the eager noise floor)
  python outputs/portfolio/bsp/bench_plan.py --save-ref
  # 2. try a variant
  python outputs/portfolio/bsp/bench_plan.py --mode compile_step
  # normal SmolVLA 80k (chunk 50) with RTC guidance, as the rollout
  python outputs/portfolio/bsp/bench_plan.py --model smolvla80k --rtc

Modes:
  eager         LeRobot code as is (bf16 VLM, as run_bsp_parcel.py)
  compile_step  torch.compile(denoise_step, mode=--compile-mode)
  static_cache  read-only prefix K/V buffers, then the same compile
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "parcel_task"))
import offline_eval as oe  # noqa: E402
from fast_denoise import enable  # noqa: E402
from segment import predict_segment  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

FRAMES = [500, 5000, 20000, 45000, 80000]  # A1 global frame indices


def load_model(name, steps, rtc):
    m = oe.Model(name, "last")
    for prm in m.policy.model.vlm_with_expert.vlm.parameters():
        if prm.dtype == torch.float32:
            prm.data = prm.data.to(torch.bfloat16)
    m.policy.config.num_steps = steps
    if rtc:
        m.policy.config.rtc_config = RTCConfig(execution_horizon=10)
        m.policy.init_rtc_processor()
    return m


def make_plan(m, rtc):
    """One plan as the robot runs it: BSP executor, sync or RTC engine."""
    if m.bsp is not None:
        def plan(o, noise):
            with torch.inference_mode():
                return predict_segment(m.policy, o, noise.clone())
        return plan
    kw = {}

    def plan(o, noise):
        ctx = torch.no_grad() if rtc else torch.inference_mode()
        with ctx:
            return m.policy.predict_action_chunk(o, noise=noise.clone(),
                                                 **kw)
    if rtc:  # RTC engine: delay ~10 ticks, leftover of the last chunk
        prev = plan(inputs.obs[0], inputs.noise)[0, 20:]
        kw.update(inference_delay=10, prev_chunk_left_over=prev)
    return plan


def inputs(m):
    ds = LeRobotDataset("Low101/parcel-sort-a",
                        root=oe.LOW / "parcel-sort-a")
    obs = []
    for i in FRAMES:
        b = torch.utils.data.default_collate([ds[i]])
        obs.append(m.pre({k: v for k, v in b.items()
                          if not k.startswith("action")}))
    cfg = m.policy.config
    gen = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim),
                        device="cuda", generator=gen)
    inputs.obs, inputs.noise = obs, noise
    return obs, noise


def plan_all(plan, obs, noise):
    return torch.cat([plan(o, noise).float() for o in obs])


def bench(plan, obs, noise, warmup, repeats):
    for _ in range(warmup):
        plan(obs[0], noise)
    torch.cuda.synchronize()
    ts = []
    for r in range(repeats):
        o = obs[r % len(obs)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        plan(o, noise)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return np.array(ts) * 1000


def diff(a, b):
    """Max |a - b| over the rows, first 7 columns (normalized units)."""
    return (a[:, :, :7] - b[:, :, :7]).abs().amax().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla_bsp",
                    choices=["smolvla_bsp", "smolvla", "smolvla80k"])
    ap.add_argument("--rtc", action="store_true",
                    help="normal SmolVLA: RTC guidance (rollout "
                    "--inference.type=rtc)")
    ap.add_argument("--freeze", action="store_true",
                    help="requires_grad off for all parameters (RTC "
                    "guidance then records no autograd graph)")
    ap.add_argument("--mode", default="eager",
                    choices=["eager", "compile_step", "static_cache"])
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="default | reduce-overhead | max-autotune | "
                         "none (static_cache only: no compile)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--save-ref", action="store_true")
    args = ap.parse_args()

    m = load_model(args.model, args.steps, args.rtc)
    if args.freeze:
        m.policy.requires_grad_(False)
    obs, noise = inputs(m)
    plan = make_plan(m, args.rtc)
    core = m.policy.model
    ref_path = HERE / (f"bench_plan_ref_{args.model}_s{args.steps}"
                       f"{'_rtc' if args.rtc else ''}.pt")

    if args.mode != "eager":
        t = time.perf_counter()
        if args.mode == "compile_step":
            core.denoise_step = torch.compile(core.denoise_step,
                                              mode=args.compile_mode)
        else:
            enable(core, args.compile_mode)
        plan_all(plan, obs, noise)  # first call compiles
        print(f"compile + first plans: {time.perf_counter() - t:.1f} s")

    ts = bench(plan, obs, noise, args.warmup, args.repeats)
    out = plan_all(plan, obs, noise)
    tag = f"{args.model}{' rtc' if args.rtc else ''}" \
        f"{' freeze' if args.freeze else ''} {args.mode}"
    if args.mode != "eager":
        tag += f" ({args.compile_mode})"
    print(f"{tag} steps={args.steps}: plan p50 "
          f"{np.median(ts):.1f} ms, p95 {np.percentile(ts, 95):.1f} ms, "
          f"min {ts.min():.1f} ms, max GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

    if args.save_ref:
        again = plan_all(plan, obs, noise)
        torch.save({"out": out.cpu()}, ref_path)
        print(f"saved {ref_path.name}; eager noise floor (same code "
              f"twice): {diff(out, again):.2e}")
    elif ref_path.exists():
        ref = torch.load(ref_path)["out"]
        print(f"max |out - eager ref|: {diff(out.cpu(), ref):.2e}"
              " (normalized units)")


if __name__ == "__main__":
    main()
