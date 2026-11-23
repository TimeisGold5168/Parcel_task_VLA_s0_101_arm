"""Live focus guide for the wrist camera.

Measures image sharpness (variance of the Laplacian) in a fixed region
and tells you whether your last turn of the lens ring helped.

Modes:
  sharp  find the sharpest focus. Turn the ring slowly; follow the text.
  train  match the focus of the training dataset, so rollouts see the
         same image quality as the data the policies learned from.

The default region is the gripper fingers. They are always in the same
place in the wrist view, so the score does not depend on the scene.
Use --roi center to focus on the area where the cube is at grasp time.

Usage:
  python -m parcel_vla.setup.focus_guide
  python -m parcel_vla.setup.focus_guide --mode train
  python -m parcel_vla.setup.focus_guide --roi center

Keys: q = quit, r = reset best score, s = save a snapshot.
"""

import argparse
import collections
import json
import os
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from parcel_vla import paths

ROOT = paths.LOW
CACHE = paths.RUNS / "camcheck/wrist_focus_ref.json"
# (y0, y1, x0, x1) in a 640x480 frame
ROIS = {"fingers": (330, 480, 160, 520), "center": (150, 330, 200, 440)}
TREND_REL = 0.04   # relative change that counts as better / worse
PEAK_REL = 0.92    # within this fraction of the best score = at peak
TRAIN_TOL = 0.25   # +-25 % of the training median = match


def sharpness(bgr, roi):
    y0, y1, x0, x1 = roi
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    # half scale: less sensitive to noise and video compression
    g = cv2.resize(g.astype(np.float32), None, fx=0.5, fy=0.5,
                   interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def training_reference(dataset, roi_name, n_eps=16, per_ep=4):
    """Median sharpness of training wrist frames, cached as JSON."""
    key = f"{dataset}:{roi_name}"
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if key in cache:
        return cache[key]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("measuring training frames (one time, about 10 s)...")
    ds = LeRobotDataset(paths.dataset_id(dataset), root=ROOT / dataset)
    eps = ds.meta.episodes
    rng = np.random.default_rng(0)
    vals = []
    for e in np.linspace(0, len(eps["dataset_from_index"]) - 1,
                         n_eps).astype(int):
        a = int(eps["dataset_from_index"][e])
        b = int(eps["dataset_to_index"][e])
        for i in rng.integers(a, b, per_ep):
            img = ds[int(i)]["observation.images.wrist"]
            rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            vals.append(sharpness(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  ROIS[roi_name]))
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    cache[key] = {"median": round(float(p50), 1),
                  "p10": round(float(p10), 1), "p90": round(float(p90), 1)}
    os.makedirs(CACHE.parent, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


class Guide:
    """Smooths the score and turns it into an instruction."""

    def __init__(self, mode, ref=None):
        self.mode, self.ref = mode, ref
        self.ema = None
        self.best = 0.0
        self.hist = collections.deque(maxlen=15)  # about 0.5 s

    def reset(self):
        self.best = 0.0

    def update(self, score):
        self.ema = score if self.ema is None else 0.7 * self.ema + 0.3 * score
        self.hist.append(self.ema)
        self.best = max(self.best, self.ema)
        old = self.hist[0]
        change = (self.ema - old) / max(old, 1e-6)
        if self.mode == "train":
            return self._train(change)
        return self._sharp(change)

    def _trend(self, change):
        if change > TREND_REL:
            return "getting SHARPER"
        if change < -TREND_REL:
            return "getting BLURRIER"
        return "steady"

    def _sharp(self, change):
        green, red, amber = (0, 200, 0), (0, 0, 255), (0, 170, 255)
        if change > TREND_REL:
            return "SHARPER - keep turning the same way", amber
        if change < -TREND_REL:
            return "BLURRIER - turn back the other way", red
        if self.ema >= PEAK_REL * self.best:
            return "AT PEAK - stop here", green
        return "past the peak - turn back slowly", red

    def _train(self, change):
        target = self.ref["median"]
        ratio = self.ema / target
        trend = self._trend(change)
        if abs(ratio - 1) <= TRAIN_TOL:
            return f"MATCHES TRAINING ({trend})", (0, 200, 0)
        if ratio > 1:
            return (f"too SHARP x{ratio:.1f} - blur it ({trend})",
                    (0, 170, 255))
        return f"too BLURRY x{ratio:.2f} - sharpen ({trend})", (0, 0, 255)


def render(frame, roi, guide, text, color):
    view = frame.copy()
    y0, y1, x0, x1 = roi
    cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
    cv2.rectangle(view, (0, 0), (640, 62), (0, 0, 0), -1)
    cv2.putText(view, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2)
    info = f"score {guide.ema:6.1f}   best {guide.best:6.1f}"
    if guide.ref:
        info += f"   training {guide.ref['median']:.1f}"
    cv2.putText(view, info, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    # bar: score relative to the best score so far (or the target)
    top = max(guide.best, guide.ref["median"] * 2 if guide.ref else 0)
    w = int(620 * min(guide.ema / max(top, 1e-6), 1.0))
    cv2.rectangle(view, (10, 470), (10 + w, 478), color, -1)
    if guide.ref:
        tx = int(10 + 620 * guide.ref["median"] / max(top, 1e-6))
        cv2.line(view, (tx, 462), (tx, 479), (255, 255, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video4")
    ap.add_argument("--mode", choices=["sharp", "train"], default="sharp")
    ap.add_argument("--roi", choices=list(ROIS), default="fingers")
    ap.add_argument("--dataset", default="third-act-attempt")
    args = ap.parse_args()

    ref = None
    if args.mode == "train":
        ref = training_reference(args.dataset, args.roi)
        print(f"training sharpness: median {ref['median']}  "
              f"(p10 {ref['p10']}, p90 {ref['p90']})")
    roi = ROIS[args.roi]
    guide = Guide(args.mode, ref)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Turn the lens ring slowly and follow the text; q to quit.")

    # opencv-python-headless has no GUI, so display with tkinter
    root = tk.Tk()
    root.title(f"wrist focus - {args.mode} (q quit, r reset, s snapshot)")
    label = tk.Label(root)
    label.pack()
    state = {"view": None}

    def save(_event=None):
        if state["view"] is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = paths.RUNS / f"camcheck/wrist_focus_{stamp}.jpg"
        os.makedirs(out.parent, exist_ok=True)
        cv2.imwrite(str(out), state["view"])
        print(f"saved {out}  score {guide.ema:.1f}")

    def update():
        ok, frame = cap.read()
        if ok:
            text, color = guide.update(sharpness(frame, roi))
            state["view"] = render(frame, roi, guide, text, color)
            rgb = cv2.cvtColor(state["view"], cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            label.configure(image=photo)
            label.image = photo
        root.after(30, update)

    root.bind("q", lambda _e: root.destroy())
    root.bind("r", lambda _e: guide.reset())
    root.bind("s", save)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    update()
    root.mainloop()
    cap.release()
    if guide.ema is not None:
        print(f"final score {guide.ema:.1f}")


if __name__ == "__main__":
    main()
