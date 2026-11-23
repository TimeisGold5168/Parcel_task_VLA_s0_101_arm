"""Skill planner: a VLM looks at the tray, code applies the priority.

At home, the planner asks the VLM yes/no questions about the top and
side camera images (one forward pass per question, P(yes) from the
"Yes"/"No" logits). Then a fixed priority picks the skill:

  stack > far_edge > box > bag; tray empty -> home

The VLM only describes the scene; the priority is code, so it is exact.

Offline test on labelled home frames (planner/labels.csv):
  python -m parcel_vla.planner.vlm_planner \
    --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
Writes runs/planner/results_<model>.csv and prints the accuracy.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from parcel_vla import paths

HERE = Path(__file__).parent
# fixed camera crops (left, top, right, bottom as a share of the image):
# the tray only, so a small VLM does not look at the floor and the desk
TOP_CROP = (0.0, 0.0, 0.45, 0.72)
SIDE_CROP = (0.25, 0.0, 1.0, 0.75)
QUESTIONS = {
    "stack": "Look at the parcels on the yellow tray. Is one box stacked "
             "on top of another box?",
    "far": "Is any parcel touching the far edge of the yellow tray, the "
           "edge farthest from the robot arm?",
    "box": "Is there a box, a rigid cube-shaped parcel, on the yellow "
           "tray?",
    "bag": "Is there a soft plastic mailer bag, white or black, on the "
           "yellow tray?",
    "tray_empty": "Is the yellow tray empty, with no parcels on it?",
}
PRIORITY = [("stack", "stack"), ("far", "far_edge"), ("box", "box"),
            ("bag", "bag")]
TASKS = {
    "stack": "pick the top box off the stack",
    "far_edge": "pick the parcel at the far edge",
    "box": "place the box in the flip zone",
    "bag": "place the bag in the flip zone",
    "home": "go home",
}


def crop(img, box):
    w, h = img.size
    return img.crop((int(box[0] * w), int(box[1] * h),
                     int(box[2] * w), int(box[3] * h)))


class VLMPlanner:
    def __init__(self, model_id, device="cuda", split=False,
                 dtype=torch.bfloat16):
        self.proc = AutoProcessor.from_pretrained(model_id)
        # no tiles: one image = one tile (far fewer tokens, ~10x faster)
        self.proc.image_processor.do_image_splitting = split
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        tok = self.proc.tokenizer
        self.yes = {tok.encode(w, add_special_tokens=False)[0]
                    for w in ("Yes", " Yes", "yes", " yes")}
        self.no = {tok.encode(w, add_special_tokens=False)[0]
                   for w in ("No", " No", "no", " no")}

    @torch.no_grad()
    def p_yes(self, images, question):
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text",
                        "text": question + " Answer yes or no."})
        msgs = [{"role": "user", "content": content}]
        inp = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        logits = self.model(**inp).logits[0, -1].float()
        y = torch.logsumexp(logits[list(self.yes)], 0)
        n = torch.logsumexp(logits[list(self.no)], 0)
        return float(torch.sigmoid(y - n))

    def observe(self, top, side):
        images = [crop(top, TOP_CROP), crop(side, SIDE_CROP)]
        return {k: self.p_yes(images, q) for k, q in QUESTIONS.items()}


def decide(obs, thr=0.5):
    if obs["tray_empty"] >= thr:
        return "home"
    for key, skill in PRIORITY:
        if obs[key] >= thr:
            return skill
    return "home"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", action="store_true",
                    help="image tiles (slow, more detail)")
    args = ap.parse_args()
    lab = pd.read_csv(HERE / "labels.csv", dtype={"id": str})
    lab = lab[lab.skip == 0]
    if args.limit:
        lab = lab.head(args.limit)
    planner = VLMPlanner(args.model, split=args.split)
    rows = []
    for _, r in lab.iterrows():
        top = Image.open(HERE / "frames" / f"{r.id}_top.jpg")
        side = Image.open(HERE / "frames" / f"{r.id}_side.jpg")
        t0 = time.perf_counter()
        obs = planner.observe(top, side)
        dt = time.perf_counter() - t0
        rows.append({"id": r.id, **{f"p_{k}": round(v, 3)
                                    for k, v in obs.items()},
                     "pred": decide(obs), "skill": r.skill, "s": dt})
        print(r.id, rows[-1]["pred"], r.skill, f"{dt:.2f}s", flush=True)
    res = pd.DataFrame(rows).merge(lab, on=["id", "skill"])
    name = args.model.split("/")[-1] + ("_split" if args.split else "")
    out = paths.RUNS / "planner"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"results_{name}.csv", index=False)
    print(f"\n{name}: {len(res)} frames, "
          f"{res.s.mean():.2f} s per decision (5 questions)")
    for k in QUESTIONS:
        ok = res[k].notna()
        acc = ((res.loc[ok, f"p_{k}"] >= 0.5) == (res.loc[ok, k] == 1))
        pos = (res.loc[ok, k] == 1).sum()
        print(f"  {k:10s} accuracy {acc.mean() * 100:5.1f} % "
              f"({ok.sum()} frames, {pos} yes)")
    ok = res.skill.notna()
    print(f"  skill      accuracy "
          f"{(res.pred[ok] == res.skill[ok]).mean() * 100:5.1f} % "
          f"({ok.sum()} frames)")
    print(pd.crosstab(res.skill[ok], res.pred[ok],
                      rownames=["true"], colnames=["pred"]))


if __name__ == "__main__":
    main()
