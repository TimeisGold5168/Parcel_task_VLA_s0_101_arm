"""Early offline check of an all-skill fine-tune checkpoint.

Scores one checkpoint and compares it with smolvla_b (old skill) and
smolvla_b_fb 10k (start of the run). Pass (exit 0) if all are true:
  1. leaves home: gripper opens >= 0.9 at 1.6 s, roll turn >= 20 deg
     (smolvla_b_fb 10k: 21.1; smolvla_b_fbs 18.3 failed the normal heap)
  2. old skill kept: B fit <= smolvla_b B fit + 1 deg
  3. new skills learned: mean fit on the black-box and on-table
     corrections < the same mean of smolvla_b_fb 10k
  4. box skill kept: box fit <= smolvla_b_fb 10k box fit + 1 deg
Else exit 1. The reference scores are saved once in
outputs/train/early_check_refs.json.

  python -m parcel_vla.finetuning.early_check smolvla_b_all:010000
"""

import json
import sys

import numpy as np
import torch

from parcel_vla.evaluation import box_fit_test as bf
from parcel_vla.evaluation import offline_eval as oe
from parcel_vla.evaluation import offline_eval_b as ob
from parcel_vla.finetuning import pick_checkpoint as pc
from parcel_vla.finetuning import start_test as stt

SETS = {"bb": pc.BB, "ot": pc.OT, "box": pc.BOX, "b": pc.B}
REFS = ob.TRAIN / "early_check_refs.json"


def scores(name):
    spec, _, step = name.partition(":")
    run, last = ob.SPECS[spec]
    oe.MODELS[spec] = ob.TRAIN / run
    m = oe.Model(spec, step or last)
    st = stt.scores(m)
    out = {"roll": float(st["old run, start frame"][0]),
           "grip": float(st["old run, 1.6 s"][1])}
    for key, (ds_name, every) in SETS.items():
        _, res = bf.fit(m, ds_name, every)
        out[key] = float(np.nanmean(np.abs(res[15])))
    del m
    torch.cuda.empty_cache()
    return out


def main():
    name = sys.argv[1]
    refs = json.loads(REFS.read_text()) if REFS.exists() else {}
    for ref in ("smolvla_b", "smolvla_b_fb:010000"):
        if ref not in refs:
            refs[ref] = scores(ref)
            REFS.write_text(json.dumps(refs, indent=1))
    old, fb = refs["smolvla_b"], refs["smolvla_b_fb:010000"]
    s = scores(name)
    new = (s["bb"] + s["ot"]) / 2
    new_fb = (fb["bb"] + fb["ot"]) / 2
    rules = {
        "1 leaves home": s["grip"] >= 0.9 and s["roll"] >= 20.0,
        "2 old skill": s["b"] <= old["b"] + 1.0,
        "3 new skills": new < new_fb,
        "4 box kept": s["box"] <= fb["box"] + 1.0,
    }
    print(json.dumps({"checkpoint": name, "scores": s,
                      "new_mean": new, "fb_new_mean": new_fb,
                      "smolvla_b": old, "smolvla_b_fb_10k": fb,
                      "rules": rules}, indent=1))
    sys.exit(0 if all(rules.values()) else 1)


if __name__ == "__main__":
    main()
