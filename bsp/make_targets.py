"""Fit B-splines to every training episode and save per-frame targets.

Usage:
  python outputs/portfolio/bsp/make_targets.py --scan
  python outputs/portfolio/bsp/make_targets.py --max-error 0.5

--scan reports fit error and compression for several tolerances.
Otherwise writes bsp/targets_<dataset>_e<err>.npz with:
  targets (n_frames, chunk_size + 2 * degree, 1 + action_dim), indexed by
  the dataset global frame index; plus degree, chunk_size, max_error.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bspline import episode_targets, make_spline, segment_range  # noqa


ROOT = Path.home() / ".cache/huggingface/lerobot/Low101"
HERE = Path(__file__).parent


def load_actions(dataset):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((ROOT / dataset / "data").rglob("*.parquet"))
    t = pa.concat_tables(
        [pq.read_table(f, columns=["index", "episode_index", "action"])
         for f in files])
    idx = np.asarray(t.column("index").to_pylist())
    ep = np.asarray(t.column("episode_index").to_pylist())
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False))
    order = np.argsort(idx)
    return idx[order], ep[order], act[order].astype(np.float64)


def build(act, ep, degree, chunk, max_error):
    out = np.zeros((len(act), chunk + 2 * degree, 1 + act.shape[1]),
                   np.float32)
    errs, ratios = [], []
    for e in np.unique(ep):
        sel = np.flatnonzero(ep == e)
        tg, err, n_knots = episode_targets(act[sel], degree, chunk,
                                           max_error)
        out[sel] = tg
        errs.append(err)
        ratios.append(len(sel) / max(n_knots, 1))
    return out, np.array(errs), np.array(ratios)


def replay_error(act, ep, targets, degree):
    """Rebuild each episode from its own targets (segment by segment)."""
    worst = []
    for e in np.unique(ep)[:20]:
        sel = np.flatnonzero(ep == e)
        rec = np.zeros_like(act[sel])
        for i in range(len(sel)):
            p = targets[sel[i]]
            t0, t1 = segment_range(p, degree)
            rec[i] = make_spline(p, degree)(np.clip(0.0, t0, t1 - 1e-6))
        worst.append(np.abs(rec - act[sel]).max(axis=0))
    return np.max(worst, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="third-act-attempt")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--max-error", type=float, default=0.5)
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    idx, ep, act = load_actions(args.dataset)
    assert (idx == np.arange(len(idx))).all(), "global index not dense"
    print(f"{args.dataset}: {len(act)} frames, {len(np.unique(ep))} "
          f"episodes, action dim {act.shape[1]}")
    errors = [0.25, 0.5, 1.0, 2.0] if args.scan else [args.max_error]
    for me in errors:
        t0 = time.time()
        tg, errs, ratios = build(act, ep, args.degree, args.chunk, me)
        rows = args.chunk + 2 * args.degree
        d = args.degree
        span = tg[::97, -(d + 1), 0] - tg[::97, d, 0]
        print(f"max_error {me}: fit error max {errs.max():.2f} "
              f"(median {np.median(errs):.2f}), frames per knot "
              f"{np.median(ratios):.1f}, segment span {np.median(span):.0f}"
              f" frames ({np.median(span) / 30:.2f} s), {rows} rows, "
              f"{time.time() - t0:.0f} s")
        print("  replay error per joint (deg):",
              np.round(replay_error(act, ep, tg, args.degree), 2))
        if not args.scan:
            out = HERE / f"targets_{args.dataset}_e{me}.npz"
            np.savez_compressed(out, targets=tg, degree=args.degree,
                                chunk_size=args.chunk, max_error=me,
                                fps=30)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
