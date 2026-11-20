"""Language test for the skill-text model (offline, held-out episodes).

    python lang_test.py smolvla_parcel_b_sk3_10k:010000 \
        smolvla_parcel_b_all_20k:010000 --out outputs/train/skill3_lang.md \
        [--split outputs/train/skill3d_split.json]

Each model is <run folder>:<checkpoint step>. Frames: the held-out
episodes of the split file (default skill3_split.json; its texts are
the texts of the test), from the start to 0.5 s
after the grasp (the part where the skill is selected), every 10
frames. For each frame, the model predicts the next 1 s with each of
the 3 skill texts and the same noise. The correct text is also run
with a second noise seed (noise floor).

Scores for each group of frames:
  error   mean |prediction - human action| of the 5 arm joints (deg)
  wins    frames where the correct text has the lowest error
  shift   mean |prediction(other text) - prediction(correct text)|,
          against the noise floor (same text, other seed)
  grip    mean of the highest predicted gripper value in the 1 s
          (a box grasp opens wider than a bag grasp)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import offline_eval as oe  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SPLIT = oe.TRAIN / "skill3_split.json"
GROUPS = {"b_box": "box", "b_bag": "bag", "stack": "top",
          "black_box": "box", "far_edge": "far", "on_table": "push"}
H, EVERY, AFTER = oe.H, 10, 15


def grasp_frame(grip):
    """First frame where the gripper closes 10 below its highest
    value after it opens above 25 (on the action)."""
    opened = np.flatnonzero(grip > 25)
    if not len(opened):
        return len(grip)
    hi = grip[opened[0]]
    for i in range(opened[0], len(grip)):
        hi = max(hi, grip[i])
        if grip[i] < hi - 10:
            return i
    return len(grip)


def test_frames(acts, split):
    rows = []
    for group, eps in split["held_out"].items():
        for e in eps:
            ep = acts[acts.episode_index == e]
            act = np.stack(ep.action.values)
            n, g = len(ep), grasp_frame(act[:, 5])
            opened = np.flatnonzero(act[:, 5] > 25)
            o = opened[0] if len(opened) else n
            for f in range(0, min(g + AFTER, n - H), EVERY):
                rows.append((int(ep["index"].iloc[f]), group, e, f,
                             "start" if f < o else "approach"))
    return pd.DataFrame(rows, columns=["index", "group", "episode",
                                       "frame", "phase"])


def run(spec, ds, act, frames, texts, batch_size):
    run_dir, _, step = spec.partition(":")
    oe.MODELS[run_dir] = oe.TRAIN / run_dir
    model = oe.Model(run_dir, step or "last")
    keys = list(texts) + ["noise"]
    preds = {k: [] for k in keys}
    for i in range(0, len(frames), batch_size):
        idx = frames["index"].values[i:i + batch_size]
        batch = torch.utils.data.default_collate([ds[int(j)] for j in idx])
        correct = [texts[GROUPS[g]] for g in
                   frames["group"].values[i:i + batch_size]]
        for k in keys:
            batch["task"] = correct if k == "noise" else [texts[k]] * len(idx)
            torch.manual_seed(i + (100000 if k == "noise" else 0))
            preds[k].append(model.predict(dict(batch)))
    del model
    torch.cuda.empty_cache()
    gt = np.stack([act[j:j + H] for j in frames["index"].values])
    return {k: np.concatenate(v) for k, v in preds.items()}, gt


def score(frames, preds, gt, texts):
    rows = []
    err = {k: np.abs(preds[k][..., :5] - gt[..., :5]).mean((1, 2))
           for k in texts}
    for (group, phase), sub in list(frames.groupby(["group", "phase"])) + \
            [((g, "all"), s) for g, s in frames.groupby("group")]:
        m = frames.index.isin(sub.index)
        right = GROUPS[group]
        best = np.argmin(np.stack([err[k][m] for k in texts]), 0)
        r = {"group": group, "phase": phase, "frames": int(m.sum()),
             "wins": float(np.mean(np.array(list(texts))[best] == right))}
        base = preds[right][m][..., :5]
        r["noise"] = float(np.abs(preds["noise"][m][..., :5] - base).mean())
        for k in texts:
            r[f"err_{k}"] = float(err[k][m].mean())
            r[f"grip_{k}"] = float(preds[k][m][..., 5].max(1).mean())
            if k != right:
                r[f"shift_{k}"] = float(
                    np.abs(preds[k][m][..., :5] - base).mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["group", "phase"])


def report(results, texts, path):
    lines = ["# Language test: skill texts (offline, held-out episodes)",
             "", "Texts: " + "; ".join(f"{k} = \"{v}\""
                                       for k, v in texts.items()), "",
             "Error and shift in deg (5 arm joints, next 1 s). Grip: "
             "highest predicted gripper value in the 1 s.", ""]
    names = " / ".join(texts)
    for spec, df in results.items():
        lines += [f"## `{spec}`", "",
                  "| Group | Phase | Frames | Correct text | Wins | "
                  f"Error {names} | Shift (other texts) | Noise | "
                  f"Grip {names} |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            right = GROUPS[r.group]
            shift = ", ".join(f"{k} {r[f'shift_{k}']:.2f}" for k in texts
                              if k != right)
            lines.append(
                f"| {r.group} | {r.phase} | {r.frames} | {right} | "
                f"{r.wins:.0%} | "
                + " / ".join(f"{r[f'err_{k}']:.2f}" for k in texts)
                + f" | {shift} | {r.noise:.2f} | "
                + " / ".join(f"{r[f'grip_{k}']:.0f}" for k in texts) + " |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default=str(oe.TRAIN / "skill3_lang.md"))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    split = json.loads(Path(a.split).read_text())
    texts = split["texts"]
    root = oe.LOW / split["dataset"]
    acts = oe.load_actions(root)
    act = np.stack(acts.action.values)
    frames = test_frames(acts, split)
    ds = LeRobotDataset("Low101/" + split["dataset"], root=root)
    results = {}
    for spec in a.models:
        preds, gt = run(spec, ds, act, frames, texts, a.batch_size)
        df = score(frames, preds, gt, texts)
        stem = Path(a.out).with_suffix("")
        df.to_csv(f"{stem}_{spec.replace(':', '_')}.csv", index=False)
        results[spec] = df
        report(results, texts, a.out)
        print(df.to_string(index=False), flush=True)
    print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
