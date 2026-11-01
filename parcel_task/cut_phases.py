"""Find the skill phases in parcel episodes from the hand-over pauses.

Episode flow: home > push > pause > flip > pause > exit push > home.
If the label is down there is one longer check pause and no flip; the
check pause itself becomes the flip segment (the robot holds still).

A pause is a run of frames where the leader action moves slower than
--still (units/s, joint units of the dataset) for at least --min-pause
seconds, away from the home pose. Output: segments.csv (one row per
segment, with its task text) and review.png (speed per episode with
the cuts) for a manual check.

If <root>/phase_marks.csv (from record_preview.py) has marks for an
episode, the marks replace the pauses: 2 marks = flip (push | flip |
exit), 1 mark = noflip (push | exit), other counts = check.

  python outputs/portfolio/parcel_task/cut_phases.py \
    --root ~/.cache/huggingface/lerobot/Low101/parcel-sort-a
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASKS = {
    "push": "push the parcel to the flip zone",
    "flip": "make the label of the parcel face down",
    "exit": "push the parcel off the table",
}
COLOURS = {"push": "#f4c430", "flip": "#1f6fd1", "exit": "#d9534f"}


def load(root):
    files = sorted((root / "data").rglob("*.parquet"))
    cols = ["episode_index", "frame_index", "timestamp", "action"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    return d.sort_values(["episode_index", "frame_index"])


def runs(mask):
    """Return (start, end) index pairs of True runs, end exclusive."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    edge = np.flatnonzero(np.diff(m))
    return list(zip(edge[::2], edge[1::2]))


def speed(act, fps, smooth):
    """Arm speed (joints 0-4 plus gripper), smoothed, units/s."""
    v = np.linalg.norm(np.diff(act, axis=0), axis=1) * fps
    v = np.concatenate([[v[0]], v])
    k = np.ones(smooth) / smooth
    return np.convolve(v, k, mode="same")


def find_pauses(act, fps, args):
    v = speed(act, fps, args.smooth)
    home = act[0]
    away = np.linalg.norm(act[:, :5] - home[:5], axis=1) > args.home_tol
    moving = np.flatnonzero((v > args.still) & away)
    if len(moving) == 0:
        return v, [], None
    first, last = moving[0], moving[-1]
    still = (v < args.still) & away
    pauses = [(a, b) for a, b in runs(still)
              if a > first and b < last and (b - a) / fps >= args.min_pause]
    return v, pauses, (first, last)


def segments(pauses, span, fps, args):
    """Cut an episode into push, flip and exit; return (rows, status)."""
    first, last = span
    lead = int(args.lead * fps)
    if len(pauses) >= 2:
        kind = "flip" if len(pauses) == 2 else "check"
        if len(pauses) > 2:
            pauses = sorted(sorted(pauses, key=lambda p: p[1] - p[0])[-2:])
        (a1, b1), (a2, b2) = pauses
        cuts = [(a1 + b1) // 2, (a2 + b2) // 2]
    elif len(pauses) == 1:
        a, b = pauses[0]
        kind = "noflip" if (b - a) / fps >= args.min_check else "check"
        mid = (a + b) // 2
        cuts = [min(a + lead, mid), max(b - lead, mid + 1)]
    else:
        return [], "check"
    bounds = [max(0, first - lead), *cuts, last + lead]
    rows = [(name, bounds[i], bounds[i + 1])
            for i, name in enumerate(["push", "flip", "exit"])]
    return rows, kind


def load_marks(root):
    path = root / "phase_marks.csv"
    if not path.exists():
        return {}
    m = pd.read_csv(path).sort_values(["episode", "mark"])
    return {ep: g.frame.tolist() for ep, g in m.groupby("episode")}


def mark_segments(marks, span, n, fps, args):
    """Cut an episode at its key marks; return (rows, status)."""
    lead = int(args.lead * fps)
    first, last = span or (0, n - 1)
    start, end = max(0, first - lead), min(n, last + lead)
    if len(marks) == 2:
        names, kind = ["push", "flip", "exit"], "flip"
    elif len(marks) == 1:
        names, kind = ["push", "exit"], "noflip"
    else:
        return [], "check"
    bounds = [start, *marks, end]
    return [(name, bounds[i], bounds[i + 1])
            for i, name in enumerate(names)], kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <root>/phases)")
    ap.add_argument("--still", type=float, default=12.0,
                    help="speed below this is still (units/s)")
    ap.add_argument("--min-pause", type=float, default=0.6)
    ap.add_argument("--min-check", type=float, default=1.5,
                    help="shortest single pause read as a no-flip check")
    ap.add_argument("--home-tol", type=float, default=6.0,
                    help="arm distance from home that counts as away")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--lead", type=float, default=0.3,
                    help="seconds kept around each cut")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    args = ap.parse_args()

    root = args.root.expanduser()
    out = args.out or root / "phases"
    out.mkdir(parents=True, exist_ok=True)
    data = load(root)
    fps = round(1 / np.median(np.diff(data.timestamp[:50])))
    eps = args.episodes or sorted(data.episode_index.unique())

    marks = load_marks(root)
    rows, plots = [], []
    for ep in eps:
        d = data[data.episode_index == ep]
        act = np.stack(d.action.to_numpy())
        v, pauses, span = find_pauses(act, fps, args)
        if ep in marks:
            segs, kind = mark_segments(marks[ep], span, len(act), fps, args)
            pauses = [(f, f + 1) for f in marks[ep]]
        else:
            segs, kind = segments(pauses, span, fps, args) if span else (
                [], "check")
        for name, a, b in segs:
            rows.append({"episode": ep, "phase": name, "start_frame": a,
                         "end_frame": b, "start_s": round(a / fps, 2),
                         "end_s": round(b / fps, 2), "kind": kind,
                         "task": TASKS[name]})
        plots.append((ep, v, pauses, segs, kind))
        src = "marks" if ep in marks else "pauses"
        print(f"ep {ep:3d}: {len(pauses)} {src:6s} -> {kind:6s} "
              + " ".join(f"{n} {a / fps:.1f}-{b / fps:.1f}s"
                         for n, a, b in segs))

    with open(out / "segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                           ["episode"])
        w.writeheader()
        w.writerows(rows)

    n = len(plots)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.4 * n + 0.5),
                             squeeze=False)
    for ax, (ep, v, pauses, segs, kind) in zip(axes[:, 0], plots):
        t = np.arange(len(v)) / fps
        for name, a, b in segs:
            ax.axvspan(a / fps, b / fps, color=COLOURS[name], alpha=.25)
        for a, b in pauses:
            ax.axvspan(a / fps, b / fps, color="k", alpha=.15)
        ax.plot(t, v, lw=.8, color="k")
        ax.axhline(args.still, color="g", lw=.6, ls="--")
        ax.set_ylim(0, np.percentile(v, 98) * 1.2 + 1)
        flag = "" if kind in ("flip", "noflip") else "  <- CHECK"
        ax.set_ylabel(f"ep {ep}\n{kind}{flag}", rotation=0, ha="right",
                      va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time (s); yellow push, blue flip, red exit, "
                           "grey pause")
    fig.tight_layout()
    fig.savefig(out / "review.png", dpi=90)
    kinds = pd.Series([p[4] for p in plots]).value_counts().to_dict()
    print(f"{kinds} -> {out / 'segments.csv'}, {out / 'review.png'}")


if __name__ == "__main__":
    main()
