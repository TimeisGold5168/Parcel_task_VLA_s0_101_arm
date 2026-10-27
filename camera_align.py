"""Live front-camera alignment against a recorded episode.

Shows the live camera blended with the first frame of a reference
episode, plus the measured rotation and shift. Adjust the camera until
the text turns green, then press q.

Usage:
  python outputs/portfolio/camera_align.py --device /dev/video6
  python outputs/portfolio/camera_align.py --ref-episode 0
Keys: q = quit, s = save a snapshot next to this script.

Headless check (no window), used by run_spot.sh:
  python outputs/portfolio/camera_align.py --check --ref-episode 0 \
    --out /tmp/align.json
  Prints the median of 10 frames, writes it as JSON, and exits with
  code 0 if aligned, 1 if not, 2 if the camera/reference failed.
"""

import argparse
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path.home() / ".cache/huggingface/lerobot/Low101"
CUBE_AREA = np.array([400.0, 330.0, 1.0])
OK_ROT_DEG = 1.0
OK_SHIFT_PX = 4.0


def reference_frame(dataset, episode):
    ds = LeRobotDataset("Low101/" + dataset, root=ROOT / dataset)
    idx = int(ds.meta.episodes["dataset_from_index"][episode])
    img = ds[idx]["observation.images.front"].permute(1, 2, 0).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


class Aligner:
    def __init__(self, ref):
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        # ignore the arm (top centre), which moves between frames
        self.mask = np.full(gray.shape, 255, np.uint8)
        self.mask[0:260, 220:420] = 0
        self.orb = cv2.ORB_create(4000)
        self.kp, self.des = self.orb.detectAndCompute(gray, self.mask)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def measure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, self.mask)
        if des is None:
            return None
        matches = sorted(self.bf.match(self.des, des),
                         key=lambda m: m.distance)[:400]
        if len(matches) < 20:
            return None
        p0 = np.float32([self.kp[m.queryIdx].pt for m in matches])
        p1 = np.float32([kp[m.trainIdx].pt for m in matches])
        aff, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if aff is None:
            return None
        rot = float(np.degrees(np.arctan2(aff[1, 0], aff[0, 0])))
        dx, dy = aff @ CUBE_AREA - CUBE_AREA[:2]
        return rot, float(dx), float(dy), int(inliers.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video6")
    ap.add_argument("--dataset", default="third-act-attempt")
    ap.add_argument("--ref-episode", type=int, default=130)
    ap.add_argument("--check", action="store_true",
                    help="measure once without a window, then exit")
    ap.add_argument("--out", help="JSON file for the --check result")
    args = ap.parse_args()

    ref = reference_frame(args.dataset, args.ref_episode)
    aligner = Aligner(ref)
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if args.check:
        sys.exit(check(cap, aligner, args))
    print("Adjust the front camera until the text is green; q to quit.")

    # opencv-python-headless has no GUI, so display with tkinter
    root = tk.Tk()
    root.title("front camera alignment (q = quit, s = snapshot)")
    label = tk.Label(root)
    label.pack()
    state = {"view": None}

    def save(_event=None):
        if state["view"] is None:
            return
        name = f"camcheck/alignment_snapshot_ep{args.ref_episode}.jpg"
        out = Path(__file__).parent / name
        os.makedirs(out.parent, exist_ok=True)
        cv2.imwrite(str(out), state["view"])
        print(f"saved {out}")

    def update():
        ok, frame = cap.read()
        if ok:
            state["view"] = render(ref, frame, aligner)
            rgb = cv2.cvtColor(state["view"], cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            label.configure(image=photo)
            label.image = photo
        root.after(30, update)

    root.bind("q", lambda _e: root.destroy())
    root.bind("s", save)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    update()
    root.mainloop()
    cap.release()


def check(cap, aligner, args, n_frames=10):
    """Median alignment over a few frames; returns the exit code."""
    for _ in range(5):  # let auto-exposure settle
        cap.read()
    results = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        res = aligner.measure(frame) if ok else None
        if res is not None:
            results.append(res)
    cap.release()
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device, "ref_dataset": args.dataset,
            "ref_episode": args.ref_episode}
    if len(results) < n_frames // 2:
        info.update(ok=False, error="camera unreadable or no match")
        code = 2
    else:
        rot, dx, dy, n = np.median(np.array(results), axis=0)
        good = bool(abs(rot) < OK_ROT_DEG
                    and np.hypot(dx, dy) < OK_SHIFT_PX)
        info.update(rot_deg=round(float(rot), 2),
                    dx_px=round(float(dx), 1), dy_px=round(float(dy), 1),
                    inliers=int(n), ok=good)
        code = 0 if good else 1
    print(json.dumps(info))
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=1))
    return code


def render(ref, frame, aligner):
    view = cv2.addWeighted(ref, 0.5, frame, 0.5, 0)
    res = aligner.measure(frame)
    if res is None:
        text, color = "no match - is this the front camera?", (0, 0, 255)
    else:
        rot, dx, dy, _ = res
        good = abs(rot) < OK_ROT_DEG and np.hypot(dx, dy) < OK_SHIFT_PX
        color = (0, 200, 0) if good else (0, 0, 255)
        text = (f"rot {rot:+.2f} deg  shift x {dx:+.1f} y {dy:+.1f} px"
                f"  {'OK' if good else 'ADJUST'}")
    cv2.rectangle(view, (0, 0), (640, 34), (0, 0, 0), -1)
    cv2.putText(view, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2)
    return view


if __name__ == "__main__":
    main()
