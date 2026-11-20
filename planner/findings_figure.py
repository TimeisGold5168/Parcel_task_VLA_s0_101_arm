"""Figure for the skill-text findings (portfolio page and writeup).

    python findings_figure.py

(a) top-view regions on the planner drive heap: the touching black box
and black bag make one region. (b) the policy closes the gripper
between them (drive run, episodes 0 and 2). (c) Test C: after key 3
the arm lifts the top box off the stack. Writes
parcel_task/figures/skill_text_findings.jpg.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "parcel_task"))
import bench_sheets as bs  # noqa: E402
import topview  # noqa: E402

OUT = HERE.parent / "parcel_task/figures/skill_text_findings.jpg"
DRIVE = "rollout_bench_parcel_smolvla_b_sk3d_planner_drive"
TEST_C = "rollout_bench_parcel_smolvla_b_sk3d_skilltext_c"
W, H = 420, 300


def frame(ds, ep, cam, t):
    root = bs.LOW / ds
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted((root / "meta/episodes").rglob("*.parquet")))
    return bs.frames_at(root, eps.set_index("episode_index").loc[ep],
                        cam, [t])[0]


def fit(im, box=None, top_band=0):
    if box:
        y0, y1, x0, x1 = box
        im = im[y0:y1, x0:x1]
    im = cv2.resize(im, (W, H - top_band))
    return np.vstack([np.zeros((top_band, W, 3), np.uint8), im])


def cap(im, text, sub):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (W, 50), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 22), 0, 0.6, (0, 255, 255), 2)
    cv2.putText(im, sub, (6, 44), 0, 0.5, (255, 255, 255), 1)
    return im


def main():
    tray = (0, 300, 0, 420)
    rgb = cv2.cvtColor(frame(DRIVE, 0, "top", 0.3), cv2.COLOR_BGR2RGB)
    regions = topview.draw(rgb, topview.parcels(rgb))
    tiles = [
        cap(fit(regions, tray), "(a) Top view: black box + black bag",
            "touching parcels = one region"),
        cap(fit(frame(DRIVE, 0, "top", 10.7), tray),
            "(b) Policy closes between them",
            "top camera, row near the robot"),
        cap(fit(frame(DRIVE, 0, "wrist", 10.4)),
            "(b) Wrist view at the close",
            "both parcels in front of the gripper"),
        cap(fit(frame(DRIVE, 2, "top", 15.9), tray),
            "(b) Same failure, other run", "gripper between box and bag"),
        cap(fit(frame(TEST_C, 0, "side", 18.0), (0, 272, 198, 640), 50),
            "(c) Key 3: 'top box' text",
            "side view: top box lifted off the stack"),
        cap(fit(frame(TEST_C, 0, "top", 19.0)),
            "(c) Key 3: 'top box' text",
            "top view: carried to the flip zone"),
    ]
    cv2.imwrite(str(OUT), np.vstack([np.hstack(tiles[:3]),
                                     np.hstack(tiles[3:])]))
    print(OUT)


if __name__ == "__main__":
    main()
