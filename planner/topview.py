"""Top-view planner: find the parcels in the tray, rank them, give the
skill of the first one.

Parcels: colour regions in the tray (top camera, 848x480, RGB), with
the arm at home masked out:
  orange -> box, white -> bag, black -> bag if large, else box
A white region smaller than MIN_WHITE is a label, not a parcel.
Limit: a black box that touches a black bag makes one black region
(a bag); the box is found when the bag is gone.
Rank (the operator's order): the row nearest to the robot first, then
the parcel nearest to the flip zone; = distance of the region centre
to ANCHOR (below the tray, at the flip-zone side).

  python outputs/portfolio/planner/topview.py <dataset> <episode> ...
writes planner/topview_check.jpg (ranks drawn on the start frames).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
TRAY = (0, 0, 325, 215)          # x0, y0, x1, y1 in the top image
ARM = (105, 150, 265, 245)       # arm at home
ANCHOR = (320, 330)
MIN_AREA = 700                   # px, smaller regions are noise
MIN_WHITE = 2500                 # px, smaller white regions are labels
BLACK_BAG = 6000                 # px, larger black regions are bags
# HSV ranges (OpenCV: H 0-180), measured on the tray: white bags are
# bluish (H ~103, S ~75); the orange box has the hue of the tray
# (21 / 23) but less saturation (~198 / ~250)
COLOURS = {
    "orange": [((10, 150, 140), (22, 228, 255))],
    "white": [((70, 0, 165), (140, 130, 255)), ((0, 0, 190), (180, 40, 255))],
    "black": [((0, 0, 0), (180, 255, 85))],
}
CLOSE = {"orange": 9, "white": 11, "black": 17}  # px, fills labels, dots


def parcels(rgb):
    """[(kind, colour, area, (cx, cy), (x, y, w, h))], ranked."""
    x0, y0, x1, y1 = TRAY
    hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    arm = np.zeros(hsv.shape[:2], np.uint8)
    ax0, ay0, ax1, ay1 = ARM
    arm[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = 1
    out = []
    for colour, ranges in COLOURS.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        m *= 1 - arm
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        k = np.ones((CLOSE[colour], CLOSE[colour]), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            area = int(st[i, 4])
            if area < MIN_AREA or (colour == "white" and area < MIN_WHITE):
                continue
            kind = "box" if colour == "orange" else "bag" \
                if colour == "white" else "bag" if area > BLACK_BAG \
                else "box"
            cx, cy = cen[i][0] + x0, cen[i][1] + y0
            box = (int(st[i, 0]) + x0, int(st[i, 1]) + y0, int(st[i, 2]),
                   int(st[i, 3]))
            out.append((kind, colour, area, (cx, cy), box))
    return sorted(out, key=lambda p: np.hypot(p[3][0] - ANCHOR[0],
                                              p[3][1] - ANCHOR[1]))


def draw(rgb, ranked):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:330, :480].copy()
    for r, (kind, colour, area, (cx, cy), (x, y, w, h)) in \
            enumerate(ranked, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(vis, (int(cx), int(cy)), 13, (0, 0, 255), -1)
        cv2.putText(vis, str(r), (int(cx) - 6, int(cy) + 6), 0, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(vis, kind, (x + 2, y + h - 4), 0, 0.5, (0, 255, 255), 2)
    skill = ranked[0][0] if ranked else "home"
    cv2.putText(vis, f"skill: {skill}", (5, 320), 0, 0.8, (255, 255, 0), 2)
    return vis


def main():
    sys.path.insert(0, str(HERE.parent / "parcel_task"))
    import bench_sheets as bs
    import pandas as pd
    args = sys.argv[1:]
    tiles = []
    for ds, e in zip(args[::2], args[1::2]):
        root = bs.LOW / ds
        eps = pd.concat(pd.read_parquet(f) for f in sorted(
            (root / "meta/episodes").rglob("*.parquet")))
        row = eps.set_index("episode_index").loc[int(e)]
        rgb = cv2.cvtColor(bs.frames_at(root, row, "top", [0.3])[0],
                           cv2.COLOR_BGR2RGB)
        ranked = parcels(rgb)
        print(ds, e, [(k, c, a) for k, c, a, _, _ in ranked])
        tiles.append(draw(rgb, ranked))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(HERE / "topview_check.jpg"), np.vstack(
        [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))


if __name__ == "__main__":
    main()
