"""Score the checkpoints of the new-skill models and pick one per model.

For each checkpoint: the start test (start_test.scores) and the grasp
fit (box_fit_test.fit, mean |err| at 0.5 s before the grasp) on the
new-skill demos and on dataset B. Writes a Markdown report.

Rules:
  1. leaves home: gripper opens >= 0.9 at "old run, 1.6 s" and roll
     turn >= 20 deg at "old run, start frame" (the stalled DAgger
     model: 6.5 deg; the old model: 32 deg);
  2. old skill kept: B fit error <= old model B fit error + 1 deg;
  3. of the rest: the lowest new-skill fit error.

  python -m parcel_vla.finetuning.pick_checkpoint \
    --out outputs/train/skill_report.md
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from parcel_vla.evaluation import box_fit_test as bf
from parcel_vla.evaluation import offline_eval as oe
from parcel_vla.evaluation import offline_eval_b as ob
from parcel_vla.finetuning import start_test as stt

BOX = ("parcel-skill-boxstack-cut", 1)
FE = ("rollout_dagger_far_edge_pull", 1)
FS = ("rollout_dagger_far_stack_pull", 1)
BB = ("rollout_dagger_black_box_cut", 1)
OT = ("rollout_dagger_on_table_cut", 1)
FSC = ("rollout_dagger_far_stack_cut", 1)
B = ("parcel-sort-b", 4)
# rule 1 roll gate: 20 deg of the old model's 32, scaled to the reference
# (BSP plans end after one segment, so they turn less in one plan).
# 15 deg let smolvla_b_fbs (18.3) pass; it failed the normal heap.
ROLL_FRAC = 20.0 / 32.0
# model: (SPECS name, new-skill set, candidate SPECS names with :step,
#         optional reference SPECS name, default smolvla_b)
GROUPS = {
    "box + far edge (smolvla_b_fb)": ("smolvla_b_fb", BOX, ["smolvla_b_box"]),
    "far edge only (smolvla_b_fe15)": ("smolvla_b_fe15", FE,
                                       ["smolvla_b_fe"]),
    "stacked-parcel corrections (smolvla_b_fbs)": (
        "smolvla_b_fbs", FS, ["smolvla_b_fb:010000"]),
    "box demos, kept? (smolvla_b_fbs)": (
        "smolvla_b_fbs", BOX, ["smolvla_b_fb:010000"]),
    "stacked-parcel corrections (smolvla_bsp_b_fbs)": (
        "smolvla_bsp_b_fbs", FS, [], "smolvla_bsp_b"),
    "box demos (smolvla_bsp_b_fbs)": (
        "smolvla_bsp_b_fbs", BOX, [], "smolvla_bsp_b"),
    "far-edge picks (smolvla_bsp_b_fb)": (
        "smolvla_bsp_b_fb", FE, [], "smolvla_bsp_b"),
    "box demos (smolvla_bsp_b_fb)": (
        "smolvla_bsp_b_fb", BOX, [], "smolvla_bsp_b"),
    "black-box corrections (smolvla_bsp_b_all)": (
        "smolvla_bsp_b_all", BB, [], "smolvla_bsp_b"),
    "box demos (smolvla_bsp_b_all)": (
        "smolvla_bsp_b_all", BOX, [], "smolvla_bsp_b"),
    "on-table corrections (smolvla_bsp_b_all)": (
        "smolvla_bsp_b_all", OT, [], "smolvla_bsp_b"),
}
# all-skill runs: from smolvla_b_fb 10k (all), from smolvla_b (allb)
for _spec in ("smolvla_b_all", "smolvla_b_allb"):
    for _name, _set in (("black-box corrections", BB),
                        ("box demos", BOX), ("far-edge picks", FE),
                        ("stacked-parcel corrections", FSC),
                        ("on-table corrections", OT)):
        GROUPS[f"{_name} ({_spec})"] = (_spec, _set,
                                        ["smolvla_b_fb:010000"])


def steps(spec):
    run, _ = ob.SPECS[spec]
    ck = ob.TRAIN / run / "checkpoints"
    return sorted(p.name for p in ck.glob("[0-9]*"))


def score(spec, step, new_set, new_only=False):
    """All scores of one checkpoint; new_only: only the new-skill fit."""
    run, last = ob.SPECS[spec]
    oe.MODELS[spec] = ob.TRAIN / run
    m = oe.Model(spec, step or last)
    fit = {}
    sets = (("new", new_set),) if new_only else (("new", new_set),
                                                  ("b", B))
    for key, (ds_name, every) in sets:
        _, res = bf.fit(m, ds_name, every)
        fit[key] = float(np.abs(res[15]).mean())
    if new_only:
        del m
        torch.cuda.empty_cache()
        return fit["new"]
    st = stt.scores(m)
    del m
    torch.cuda.empty_cache()
    return {"roll": st["old run, start frame"][0],
            "grip": st["old run, 1.6 s"][1],
            "reach": st["B starts (25 eps)"][2],
            "new": fit["new"], "b": fit["b"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ob.TRAIN / "skill_report.md")
    ap.add_argument("--only", default=None,
                    help="score only the groups whose title has this text")
    args = ap.parse_args()
    lines = ["# New-skill checkpoints: offline scores", "",
             "Roll: start test, old-run start frame (pass >= 20, old "
             "model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp "
             "pose error 0.5 s before the grasp, deg (lower is better).",
             ""]
    old = {}
    for title, (spec, new_set, extra, *ref) in GROUPS.items():
        if args.only and args.only not in title:
            continue
        ref = ref[0] if ref else "smolvla_b"
        cands = [f"{spec}:{s}" for s in steps(spec)] + extra + [ref]
        rows = {}
        for c in cands:
            name, _, step = c.partition(":")
            print(f"scoring {c}", flush=True)
            if c in old:
                rows[c] = dict(old[c])
                rows[c]["new"] = score(name, step, new_set, new_only=True)
                continue
            rows[c] = score(name, step, new_set)
            old[c] = rows[c]
        b_max = rows[ref]["b"] + 1.0
        roll_min = ROLL_FRAC * rows[ref]["roll"]
        ok = [c for c in cands if c != ref
              and rows[c]["grip"] >= 0.9 and rows[c]["roll"] >= roll_min
              and rows[c]["b"] <= b_max]
        pick = min(ok, key=lambda c: rows[c]["new"]) if ok else None
        lines += [f"## {title}", "",
                  f"Reference `{ref}`: roll >= {roll_min:.1f}, "
                  f"B fit <= {b_max:.1f}.", "",
                  "| Checkpoint | Roll | Grip | New-skill fit | B fit "
                  "| Passes rules 1-2 |",
                  "|---|---|---|---|---|---|"]
        for c in cands:
            r = rows[c]
            lines.append(f"| `{c}` | {r['roll']:.1f} | {r['grip']:.2f} "
                         f"| {r['new']:.1f} | {r['b']:.1f} "
                         f"| {'yes' if c in ok else 'no'} |")
        lines += ["", f"Pick: `{pick}`" if pick else
                  "Pick: none passes the rules", ""]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
