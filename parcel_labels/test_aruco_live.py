"""Live ArUco test of the printed parcel labels with a robot camera.

Opens one camera, finds DICT_4X4_100 markers in every frame and shows
the result in an ffplay window (OpenCV here has no GUI): a green box
and "ID <n> <size> cm <px> px" for each label in labels.csv, a red box
for any ID that is not in labels.csv (a false detection).

Press Ctrl+C to stop. Then it prints, per marker ID, the share of
frames with a detection and the marker side in pixels, and saves the
last annotated frame and a CSV with the per-frame detections.

Usage:
  python outputs/portfolio/parcel_labels/test_aruco_live.py
  python outputs/portfolio/parcel_labels/test_aruco_live.py \
    --device /dev/video4 --name wrist
"""

import argparse
import csv
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
FFPLAY = shutil.which("ffplay") or "ffplay"


def load_labels():
    path = HERE / "labels.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(r["marker_id"]): float(r["size_cm"])
                for r in csv.DictReader(f)}


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {device} (in use, or "
                         "USB bandwidth: see the earlier ffmpeg error)")
    return cap


def annotate(frame, corners, ids, labels, fps_now):
    for quad, mid in zip(corners, ids):
        pts = quad[0].astype(int)
        side = float(np.mean(np.linalg.norm(
            quad[0] - np.roll(quad[0], 1, axis=0), axis=1)))
        known = mid in labels
        color = (0, 200, 0) if known else (0, 0, 255)
        cv2.polylines(frame, [pts], True, color, 2)
        text = (f"ID {mid} {labels[mid]:g}cm {side:.0f}px" if known
                else f"FALSE ID {mid}")
        org = (int(pts[:, 0].min()), int(pts[:, 1].min()) - 6)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1)
    cv2.putText(frame, f"{len(ids)} markers  {fps_now:.0f} fps", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video6",
                    help="camera (front /dev/video6, wrist /dev/video4)")
    ap.add_argument("--name", default="front", help="name for outputs")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=0,
                    help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    labels = load_labels()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
    cap = open_camera(args.device, args.width, args.height, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera opened but gave no frame")
    h, w = frame.shape[:2]
    view = None
    if not args.no_window:
        view = subprocess.Popen(
            [FFPLAY, "-loglevel", "error", "-nostats", "-fflags",
             "nobuffer", "-window_title", f"aruco test: {args.name}",
             "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
             f"{w}x{h}", "-framerate", str(args.fps), "-i", "-"],
            stdin=subprocess.PIPE)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = HERE / "live_tests"
    out.mkdir(exist_ok=True)
    log_path = out / f"aruco_{args.name}_{stamp}.csv"
    seen, sides = defaultdict(int), defaultdict(list)
    n_frames, false_hits = 0, 0
    t0 = last = time.perf_counter()
    fps_now = 0.0
    print(f"testing {args.device} ({w}x{h}); press Ctrl+C to stop")
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["t_s", "marker_id", "known", "side_px",
                      "cx", "cy"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.perf_counter()
                fps_now = 0.9 * fps_now + 0.1 / max(now - last, 1e-3)
                last = now
                n_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                ids = [] if ids is None else ids.ravel().tolist()
                for quad, mid in zip(corners, ids):
                    side = float(np.mean(np.linalg.norm(
                        quad[0] - np.roll(quad[0], 1, axis=0), axis=1)))
                    cx, cy = quad[0].mean(0)
                    log.writerow([f"{now - t0:.3f}", mid,
                                  int(mid in labels), f"{side:.1f}",
                                  f"{cx:.0f}", f"{cy:.0f}"])
                    if mid in labels:
                        seen[mid] += 1
                        sides[mid].append(side)
                    else:
                        false_hits += 1
                frame = annotate(frame, corners, ids, labels, fps_now)
                if view is not None:
                    try:
                        view.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        view = None
                if args.duration and now - t0 >= args.duration:
                    break
        except KeyboardInterrupt:
            pass
    cap.release()
    if view is not None:
        view.stdin.close()
        view.terminate()
    cv2.imwrite(str(out / f"aruco_{args.name}_{stamp}.jpg"), frame)

    secs = time.perf_counter() - t0
    print(f"\n{n_frames} frames in {secs:.1f} s; "
          f"false detections: {false_hits}")
    print(f"{'ID':>4} {'size':>6} {'detected':>9} {'side px':>8}")
    for mid in sorted(seen, key=lambda m: -labels[m]):
        print(f"{mid:>4} {labels[mid]:>4g}cm "
              f"{100 * seen[mid] / max(n_frames, 1):>8.0f}% "
              f"{np.median(sides[mid]):>8.0f}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
