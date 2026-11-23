"""Inspect one recorded heap: videos, marks, schedule and demo quality.

Episodes map to heaps by the heap sizes in schedule_A1.csv (heap 1 is
the first episodes, and so on). Prints:
  - per episode: length, phase marks, missing video frames, idle start,
    stops > 1 s, flip time;
  - label up / down counts against the schedule;
  - quality of this heap against all earlier heaps;
and writes <out>/heap<N>_marks.png (top view at mark 1 of each episode)
to check the parcel kind and label by eye.

  python -m parcel_vla.evaluation.inspect_heap --heap 4
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from parcel_vla import paths

HERE = Path(__file__).parent
ROOT = paths.LOW / "parcel-sort-a"


def load(root):
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted((root / "data").rglob("*.parquet"))])
    eps = pd.concat([pd.read_parquet(f) for f in
                     sorted((root / "meta/episodes").rglob("*.parquet"))])
    marks = pd.read_csv(root / "phase_marks.csv")
    return data, eps.set_index("episode_index"), marks


def heap_ranges(schedule):
    sizes = schedule.groupby("block").size()
    ends = sizes.cumsum()
    return {b: range(ends[b] - sizes[b], ends[b]) for b in sizes.index}


def quality(act, fps, marks):
    """Idle start, stops > 1 s and flip time of one episode (seconds)."""
    v = np.linalg.norm(np.diff(act[:, :5], axis=0), axis=1) * fps
    v = np.convolve(v, np.ones(5) / 5, "same")
    moving = np.flatnonzero(v > 8)
    if moving.size == 0:
        return {"idle_s": len(act) / fps, "stops": 0, "stop_s": 0.0,
                "longest_s": 0.0, "flip_s": np.nan}
    first, last = moving[0], moving[-1]
    stops, i = [], first
    while i < last:
        if v[i] < 3:
            j = i
            while j < last and v[j] < 3:
                j += 1
            if j - i > fps:
                stops.append((j - i) / fps)
            i = j
        else:
            i += 1
    flip = (marks[1] - marks[0]) / fps if len(marks) == 2 else np.nan
    return {"idle_s": first / fps, "stops": len(stops),
            "stop_s": sum(stops), "longest_s": max(stops, default=0.0),
            "flip_s": flip}


def mark_sheet(root, eps, marks, episodes, fps, path):
    tiles = []
    for ep in episodes:
        r = eps.loc[ep]
        m = marks.get(ep, [])
        t = r["videos/observation.images.top/from_timestamp"] + (
            m[0] / fps + 0.3 if m else min(5.0, r["length"] / fps / 2))
        f = int(r["videos/observation.images.top/file_index"])
        tile = path.parent / f"_tile_{ep}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i",
             str(root / f"videos/observation.images.top/chunk-000/"
                        f"file-{f:03d}.mp4"),
             "-frames:v", "1", "-vf",
             f"scale=420:-2,drawtext=text='ep{ep} mark 1':x=5:y=5:"
             "fontsize=22:fontcolor=yellow:box=1:boxcolor=black",
             str(tile)], check=True)
        tiles.append(tile)
    cols = 4
    while len(tiles) % cols:
        tiles.append(tiles[-1])
    layout = "|".join(
        "_".join(["+".join(["w0"] * (i % cols)) or "0",
                  "+".join(["h0"] * (i // cols)) or "0"])
        for i in range(len(tiles)))
    args = ["ffmpeg", "-v", "error", "-y"]
    for t in tiles:
        args += ["-i", str(t)]
    args += ["-filter_complex",
             f"xstack=inputs={len(tiles)}:layout={layout}", str(path)]
    subprocess.run(args, check=True)
    for t in set(tiles):
        t.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heap", type=int, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--schedule", type=Path,
                    default=HERE.parent / "data/schedule_A1.csv")
    ap.add_argument("--out", type=Path, default=Path("/tmp/appr"))
    args = ap.parse_args()

    root = args.root.expanduser()
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    data, eps, mk = load(root)
    marks = {e: g.sort_values("mark").frame.tolist()
             for e, g in mk.groupby("episode")}
    sched = pd.read_csv(args.schedule)
    ranges = heap_ranges(sched)
    episodes = [e for e in ranges[args.heap] if e in eps.index]
    if not episodes:
        raise SystemExit(f"no episodes recorded for heap {args.heap}")

    cams = sorted({c.split("/")[1] for c in eps.columns
                   if c.startswith("videos/")})
    rows = []
    for ep, g in data.groupby("episode_index"):
        heap = next(b for b, r in ranges.items() if ep in r)
        if heap > args.heap:
            continue
        act = np.stack(g.action.to_numpy())
        state = np.stack(g["observation.state"].to_numpy())
        r = eps.loc[ep]
        missing = sum(max(0, int(r.length) - round(
            (r[f"videos/{c}/to_timestamp"]
             - r[f"videos/{c}/from_timestamp"]) * fps)) for c in cams)
        rows.append({"heap": heap, "ep": ep, "len_s": len(g) / fps,
                     "marks": len(marks.get(ep, [])), "missing": missing,
                     "nan": bool(np.isnan(act).any()
                                 or np.isnan(state).any()),
                     **quality(act, fps, marks.get(ep, [])),
                     "start": state[0, :5], "end": state[-1, :5]})
    df = pd.DataFrame(rows)
    cur = df[df.heap == args.heap]

    show = ["ep", "len_s", "marks", "missing", "nan", "idle_s", "stops",
            "longest_s", "flip_s"]
    print(f"heap {args.heap}: {len(cur)} of {len(ranges[args.heap])} "
          f"episodes recorded")
    print(cur[show].round(1).to_string(index=False))

    plan = sched[sched.block == args.heap].label.value_counts()
    got_up = int((cur.marks == 2).sum())
    got_down = int((cur.marks == 1).sum())
    print(f"labels: schedule {plan.get('up', 0)} up / "
          f"{plan.get('down', 0)} down, recorded {got_up} up / "
          f"{got_down} down")
    kinds = sched[sched.block == args.heap].groupby(
        ["colour", "type", "label"]).size()
    print("schedule:", ", ".join(f"{n} {c} {t} {lab}"
                                 for (c, t, lab), n in kinds.items()))

    def summary(part):
        poses = np.stack(list(part.start) + list(part.end))
        std = poses.std(0)
        return {"idle start mean (s)": part.idle_s.mean(),
                "stops > 1 s per episode": part.stops.mean(),
                "stop time per episode (s)": part.stop_s.mean(),
                "longest stop (s)": part.longest_s.max(),
                "episodes with stop > 3 s": int((part.longest_s > 3).sum()),
                "flip mean (s)": part.flip_s.mean(),
                "flip std (s)": part.flip_s.std(),
                "home std shoulder_lift (deg)": std[1],
                "home std wrist_flex (deg)": std[3]}

    table = {f"heap {args.heap}": summary(cur)}
    prev = df[df.heap < args.heap]
    if len(prev):
        table = {f"heaps 1-{args.heap - 1}": summary(prev), **table}
    print(pd.DataFrame(table).round(1).to_string())

    bad = cur[(cur.missing > 0) | cur.nan | (cur.marks == 0)]
    print("problems:", "none" if bad.empty else
          ", ".join(f"ep {e}" for e in bad.ep))

    args.out.mkdir(parents=True, exist_ok=True)
    sheet = args.out / f"heap{args.heap}_marks.png"
    mark_sheet(root, eps, marks, episodes, fps, sheet)
    print(f"mark-1 images: {sheet}")


if __name__ == "__main__":
    main()
