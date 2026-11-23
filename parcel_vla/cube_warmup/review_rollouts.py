"""Auto-score spot-test rollouts and build review sheets.

For each episode of a rollout dataset it fills runs/results_spots.csv
columns automatically:
  stage_reached  none / reach / grasp / lift / place
  task_success   1 if place (cube released into the bin)
  pick_success   1 if lift or place (cube grasped and off the table)
  recovery       yes / no if the first grasp failed (missed the cube or
                 dropped it) and a later grasp did / did not reach the
                 bin; n/a if the first grasp worked or nothing closed
  miss_direction where the first close landed relative to the cube
                 (toward_square, away_square, left, right, short, long,
                 none = landed on the cube)
  miss_cm        distance between that landing point and the cube
  notes          close attempts, hold time, "check" if unsure
  loop_hz        recorded frames / episode time (valid for full-length
                 episodes; below 30 Hz the robot moves in slow motion)
  place_s        wall-clock time from start to the release in the bin
                 (place = gripper opens after the last hold, then a
                 small cube blob is seen in the bin)
  jerk           RMS jerk of commanded arm joints, 1000 deg/s^3
  blue_train_cm  (D / DUO only) blue cube to the nearest training start
                 position of the green cube
  blue_green_cm  (D / DUO only) blue cube to green cube
  close_frac     (D / DUO only) first close on the green -> blue line:
                 0 = green, 1 = blue, ~0.5 = between
  close_off_cm   (D / DUO only) first close distance from that line

How:
- Gripper signal: closed on nothing reads ~2; holding the cube the
  joint stops higher than the command.
- Lift: shoulder_lift rises while holding.
- Place: the green cube is seen inside the bin near the end.
- Landing point: a joints -> front-image-pixel map fitted on the
  grasp moments of the training dataset (cached in runs/spot_review/).

Also saves one contact sheet per dataset (rows = episodes):
  start | first close | 1 s after first hold | release | end

Usage:
  python -m parcel_vla.cube_warmup.review_rollouts rollout_act151_X1
  (prefix match, newest dataset). Rows in results_spots.csv for that
  model/spot are filled only where stage_reached is still empty.
  Each row is tagged src=<dataset>:ep<n> in notes; scoring the same
  dataset again updates those rows instead of adding new ones.
  RESULTS_CSV=<file> writes to another table (e.g. results_configs.csv).
"""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from parcel_vla import paths

ROOT = paths.LOW
OUT = paths.RUNS / "spot_review"
RESULTS = Path(os.environ.get("RESULTS_CSV",
                              paths.RUNS / "results_spots.csv"))
EPISODE_S = 30.0   # --dataset.episode_time_s in run_robot_test.sh
TRAIN_SET = "third-act-attempt"
CHECKPOINTS = {
    "act151": "050000",
    "act_aug": "025000",
    "act51": "100000",
    "diffusion": "100000",
    "smolvla": "100000",
    "smolvla_aug": "025000",
}
FPS = 30
GRIP, LIFT = 5, 1
CLOSE_CMD = 15.0   # commanded gripper below this = closing
HOLD_GAP = 1.5     # state above command by this = object in gripper
HOLD_MIN = 5.0     # empty gripper closes to ~2
CUBE_CM = 2.5
ON_CUBE_CM = 1.5   # landing closer than this counts as on the cube
FAR_MISS_CM = 8.0  # beyond the fitted map range: distance is rough
BIN_PX = 60        # cube centre to bin centre (training ends <= 37)
IN_BIN_AREA = 2000  # cube blob px: in the bin < 1300, held > 2700
PICKED = ("lift", "place")  # cube grasped and off the table
PALE_S_MIN = 50    # start-frame fallback; bin check keeps s_min 80
# blue distractor cube: compact square blob; the blue tape around the
# square and under the bin is thin (fill ~0.35) and is rejected
BLUE_HSV = ((95, 120, 50), (130, 255, 255))
BLUE_FILL = 0.75   # blob area / bounding-box area
BLUE_AREA = (600, 3000)
DISTRACTOR_FIELDS = ["blue_train_cm", "blue_green_cm", "close_frac",
                     "close_off_cm"]
APPROACH_MIN_S = 3.0  # ignore closes while the arm is still at home


def newest(prefix):
    cands = sorted(ROOT.glob(prefix + "*"), key=os.path.getmtime)
    if not cands:
        sys.exit(f"no dataset matching {prefix}*")
    return cands[-1].name


def load(name):
    ds = LeRobotDataset(paths.dataset_id(name), root=ROOT / name)
    hf = ds.hf_dataset.with_format("numpy")
    return (ds, np.stack(hf["observation.state"]),
            np.stack(hf["action"]), np.array(hf["episode_index"]))


def gripper_events(state, action):
    g, c = state[:, GRIP], action[:, GRIP]
    closing = c < CLOSE_CMD
    holding = closing & ((g - c) > HOLD_GAP) & (g > HOLD_MIN)
    win = FPS // 2
    run = np.convolve(holding, np.ones(win), "same") >= win * 0.6
    # keep only holds lasting at least 0.5 s
    edges = np.flatnonzero(np.diff(np.r_[0, run.astype(int), 0]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi - lo < win:
            run[lo:hi] = False
    # join holds split by a short command wobble while the cube stays in
    # the gripper (state unchanged), so only real drops count as regrasps
    if run.any():
        held_g = float(np.median(g[run]))
        edges = np.flatnonzero(np.diff(np.r_[0, run.astype(int), 0]))
        for hi, lo in zip(edges[1:-1:2], edges[2::2]):
            if lo - hi < FPS and np.all(np.abs(g[hi:lo] - held_g) < 5):
                run[hi:lo] = True
    starts = np.flatnonzero(closing[1:] & ~closing[:-1]) + 1
    first_close = int(starts[0]) if len(starts) else None
    hold_idx = np.flatnonzero(run)
    first_hold = int(hold_idx[0]) if len(hold_idx) else None
    # release = gripper opens after the last hold (the place, not a drop)
    release = None
    if first_hold is not None:
        last_hold = int(np.flatnonzero(np.diff(np.r_[0, run.astype(int)])
                                       == 1)[-1])
        after = np.flatnonzero(~closing[last_hold:])
        if len(after):
            release = last_hold + int(after[0])
    # closes after the release in the bin are not grasp attempts
    tries = starts[starts < release] if release is not None else starts
    return {
        "close_attempts": int(len(tries)),
        "first_close_s": None if first_close is None
        else round(first_close / FPS, 1),
        "held_s": round(float(run.sum()) / FPS, 1),
        "first_hold_s": None if first_hold is None
        else round(first_hold / FPS, 1),
        "release_s": None if release is None else round(release / FPS, 1),
        "holds": int(np.count_nonzero(np.diff(np.r_[0, run.astype(int)])
                                      == 1)),
        "_idx": [first_close, first_hold, release],
        "_run": run,
    }


def rgb(ds, i, key="observation.images.front"):
    img = ds[int(i)][key].permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def find_cube(img, min_area=150, s_min=80):
    """Largest green blob: (centre xy, area) or (None, 0)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (35, s_min, 60), (85, 255, 255))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, 0
    k = 1 + int(np.argmax(stats[1:, 4]))
    if stats[k, 4] < min_area:
        return None, 0
    return cents[k], int(stats[k, 4])


def find_blue_cube(img, bin_px):
    """Blue distractor cube: (centre xy, area) or (None, 0)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, *BLUE_HSV)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    best = (None, 0)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not BLUE_AREA[0] <= area <= BLUE_AREA[1]:
            continue
        if area / (w * h) < BLUE_FILL or not 0.7 <= w / h <= 1.4:
            continue
        if cents[k][1] < 100 or np.linalg.norm(cents[k] - bin_px) < 60:
            continue   # table edge / robot base, or the bin tape
        if area > best[1]:
            best = (cents[k], int(area))
    return best


def train_starts():
    """[x, y, area] of the green cube at the start of every training
    episode (cached)."""
    path = OUT / "train_starts_v2.json"
    if path.exists():
        return np.array(json.loads(path.read_text()))
    print("finding training start positions (once)...")
    ds, _, _, e = load(TRAIN_SET)
    pts = []
    for ep in np.unique(e):
        img = rgb(ds, np.flatnonzero(e == ep)[0])
        c, area = find_cube(img, 300)
        if c is not None:
            pts.append([float(c[0]), float(c[1]), area])
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(pts))
    return np.array(pts)


def px_per_cm(pt):
    """Image scale at a table point: median training cube size of the 5
    nearest training starts (perspective, and not today's lighting)."""
    train = train_starts()
    near = np.argsort(np.linalg.norm(train[:, :2] - pt, axis=1))[:5]
    return float(np.sqrt(np.median(train[near, 2])) / CUBE_CM)


def distractor(ds, s, a, sel, cal):
    """blue_train_cm, close_frac, close_off_cm for a distractor trial.

    blue_train_cm: blue cube to the nearest training start position.
    close_frac: first-close landing point projected on the line from
      the green cube (0) to the blue cube (1); ~0.5 = between them.
    close_off_cm: distance of the landing point from that line.
    Closes in the first APPROACH_MIN_S are skipped: the arm is still at
    home, in the air, and the pixel map only holds at table height.
    """
    hz = min(len(sel) / EPISODE_S, float(FPS))
    closing = a[sel, GRIP] < CLOSE_CMD
    starts = np.flatnonzero(closing[1:] & ~closing[:-1]) + 1
    starts = starts[starts >= APPROACH_MIN_S * hz]
    close = int(starts[0]) if len(starts) else None
    img = rgb(ds, sel[0])
    green, _ = find_cube(img, 300)
    if green is None:
        green, _ = find_cube(img, 300, PALE_S_MIN)
    blue, _ = find_blue_cube(img, np.array(cal["bin_px"]))
    if green is None or blue is None:
        return {f: "" for f in DISTRACTOR_FIELDS}
    train = train_starts()[:, :2]
    out = {"blue_train_cm": round(float(
        np.min(np.linalg.norm(train - blue, axis=1)) / px_per_cm(blue)),
        1),
        "blue_green_cm": round(float(np.linalg.norm(blue - green)
                                     / px_per_cm((blue + green) / 2)), 1)}
    if close is None:
        return {**out, "close_frac": "", "close_off_cm": ""}
    grip = (joint_feats(s[sel[close]][None, :4]) @ np.array(cal["w"]))[0]
    line = blue - green
    frac = float((grip - green) @ line / (line @ line))
    off = abs(float(line[0] * (grip - green)[1]
                    - line[1] * (grip - green)[0]) / np.linalg.norm(line))
    return {**out, "close_frac": round(frac, 2),
            "close_off_cm": round(off / px_per_cm(grip), 1)}


def joint_feats(j):
    r = np.radians(j)
    f = [np.ones(len(j))]
    for c in range(4):
        f += [np.sin(r[:, c]), np.cos(r[:, c])]
    f += [j[:, 0] * j[:, 1] / 1e3, j[:, 1] * j[:, 2] / 1e3,
          j[:, 0] * j[:, 2] / 1e3, j[:, 1] ** 2 / 1e3,
          j[:, 2] ** 2 / 1e3]
    return np.stack(f, 1)


def fit(feats, y):
    a = feats.T @ feats + 1e-2 * np.eye(feats.shape[1])
    return np.linalg.solve(a, feats.T @ y)


def calibration():
    """Joints -> gripper pixel map, square centre and bin centre."""
    path = OUT / "gripper_pixel_map.json"
    if path.exists():
        return json.loads(path.read_text())
    print("fitting joints -> pixel map on training grasps (once)...")
    ds, s, a, e = load(TRAIN_SET)
    joints, pix, square, bins = [], [], [], []
    for ep in np.unique(e):
        sel = np.flatnonzero(e == ep)
        hold = gripper_events(s[sel], a[sel])["_idx"][1]
        cube, _ = find_cube(rgb(ds, sel[0]), 300)
        if hold is not None and cube is not None:
            joints.append(s[sel[hold], :4])
            pix.append(cube)
            if ep <= 50:
                square.append(cube)
        end, area = find_cube(rgb(ds, sel[-1]))
        if end is not None:
            bins.append(end)
    joints, pix = np.array(joints), np.array(pix)
    w = fit(joint_feats(joints), pix)
    err = np.linalg.norm(joint_feats(joints) @ w - pix, axis=1)
    keep = err < 50
    w = fit(joint_feats(joints[keep]), pix[keep])
    cal = {"w": w.tolist(),
           "square_px": np.mean(square, 0).tolist(),
           "bin_px": np.median(bins, 0).tolist()}
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(cal, indent=1))
    return cal


def miss(cal, joints, cube, cube_area):
    """Direction and cm from cube to predicted gripper landing point."""
    grip = joint_feats(joints[None, :4]) @ np.array(cal["w"])
    v = grip[0] - cube
    px_per_cm = np.sqrt(cube_area) / CUBE_CM
    cm = float(np.linalg.norm(v) / px_per_cm)
    if cm < ON_CUBE_CM:
        return "none", round(cm, 1)
    unit = v / np.linalg.norm(v)
    to_sq = np.array(cal["square_px"]) - cube
    if np.linalg.norm(to_sq) > 2 * np.sqrt(cube_area):
        cos = float(unit @ to_sq / np.linalg.norm(to_sq))
        if cos > 0.7:
            return "toward_square", round(cm, 1)
        if cos < -0.7:
            return "away_square", round(cm, 1)
    # robot base is at the top centre of the front image
    radial = cube - np.array([320.0, 0.0])
    radial /= np.linalg.norm(radial)
    along = float(unit @ radial)
    across = float(unit[0] * radial[1] - unit[1] * radial[0])
    if abs(along) >= abs(across):
        return ("long" if along > 0 else "short"), round(cm, 1)
    return ("left" if across > 0 else "right"), round(cm, 1)


def cube_in_bin(ds, sel, cal, rel):
    """True / False, or None if the cube is hidden at the end.

    Only frames after the release count, and the blob must be small:
    a cube inside the bin is part-hidden by the rim (training ends:
    area <= 1223 px), a cube still in the gripper above the bin is
    closer to the camera (> 2700 px).
    """
    if rel is None:
        return False
    bin_px = np.array(cal["bin_px"])
    for k in range(len(sel) - 1, max(len(sel) - 3 * FPS, rel - 1), -10):
        c, area = find_cube(rgb(ds, sel[k]))
        if c is not None:
            return bool(np.linalg.norm(c - bin_px) < BIN_PX
                        and area < IN_BIN_AREA)
    return None


def score(ds, s, a, sel, cal, spot=""):
    ev = gripper_events(s[sel], a[sel])
    close, hold, rel = ev["_idx"]
    is_duo = spot.endswith("DUO")
    cube0, area0 = (None, 0) if is_duo else find_cube(rgb(ds, sel[0]), 300)
    if cube0 is None and not is_duo:
        # pale cube under bright light: saturation 70-90, below 80
        cube0, area0 = find_cube(rgb(ds, sel[0]), 300, PALE_S_MIN)
    notes = [f"attempts={ev['close_attempts']}", f"held={ev['held_s']}s"]
    if is_duo and len(ds.meta.tasks.index):
        notes.append(f"task={ds.meta.tasks.index[0]}")
    direction, cm = "", ""
    if cube0 is not None and close is not None:
        if hold is not None and abs(hold - close) < FPS:
            direction, cm = "none", 0
        else:
            direction, cm = miss(cal, s[sel[close]], cube0, area0)
    elif is_duo and close is not None:
        on_close = hold is not None and abs(hold - close) < FPS
        direction = "none" if on_close else "check"
        cm = 0 if on_close else ""
    elif cube0 is None and not is_duo:
        notes.append("check: cube not found at start")
    if cm != "" and float(cm) > FAR_MISS_CM:
        notes.append("check: far miss, cm is rough")
    in_bin = None if is_duo else cube_in_bin(ds, sel, cal, rel)
    lifted = False
    if hold is not None:
        held = ev["_run"]
        lift_vals = s[sel][held, LIFT]
        lifted = (s[sel[hold], LIFT] - lift_vals.min()) > 8
    if is_duo and lifted and rel is not None:
        grip_rel = joint_feats(s[sel[rel]][None, :4]) @ np.array(cal["w"])
        dist = np.linalg.norm(grip_rel[0] - np.array(cal["bin_px"]))
        in_bin = bool(dist < 75)
    if in_bin:
        stage = "place"
    elif lifted:
        stage = "lift"
    elif hold is not None:
        stage = "grasp"
    else:
        end, _ = (None, 0) if is_duo else find_cube(rgb(ds, sel[-1]), 300)
        moved = (cube0 is not None and end is not None
                 and np.linalg.norm(end - cube0) > 15)
        near = direction not in ("", "check") and float(cm) < 2 * ON_CUBE_CM
        stage = "reach" if moved or near else "none"
    if in_bin is None and stage != "place":
        notes.append("check: cube hidden at end")
    first_ok = (close is not None and hold is not None
                and abs(hold - close) < FPS and ev["holds"] == 1)
    if close is None or first_ok:
        recovery = "n/a"
    else:
        recovery = "yes" if stage == "place" else "no"
    if ev["holds"] > 1:
        notes.append(f"holds={ev['holds']}")
    return ev, stage, "; ".join(notes), direction, cm, recovery


def timing(a_sel, rel, stage):
    """(loop_hz, place_s, jerk) for one episode."""
    hz = min(len(a_sel) / EPISODE_S, float(FPS))
    place_s = round(rel / hz, 1) if stage == "place" and rel else ""
    arm = a_sel[:, :GRIP].astype(np.float64)
    jerk = ""
    if len(arm) > 3:
        j = np.diff(arm, 3, axis=0) * hz ** 3
        jerk = round(float(np.sqrt(np.mean(j ** 2))) / 1000, 1)
    return round(hz, 1), place_s, jerk


def tile(ds, i, label):
    front = cv2.resize(rgb(ds, i), (320, 240))
    wrist = cv2.resize(rgb(ds, i, "observation.images.wrist"), (320, 240))
    t = cv2.cvtColor(np.vstack([front, wrist]), cv2.COLOR_RGB2BGR)
    cv2.rectangle(t, (0, 0), (320, 22), (0, 0, 0), -1)
    cv2.putText(t, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return t


def blank(label):
    t = np.full((480, 320, 3), 40, np.uint8)
    cv2.putText(t, label, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return t


CAM_FIELDS = ["cam_ref_ep", "cam_rot_deg", "cam_shift_px"]
TIME_FIELDS = ["loop_hz", "place_s", "jerk"]


def camera_columns(name):
    """Camera offset recorded by run_spot.sh for this dataset."""
    path = ROOT / name / "camera_alignment.json"
    if not path.exists():
        return {}
    a = json.loads(path.read_text())
    if "rot_deg" not in a:
        return {"cam_ref_ep": a.get("ref_episode", "")}
    return {"cam_ref_ep": a["ref_episode"], "cam_rot_deg": a["rot_deg"],
            "cam_shift_px": round(float(np.hypot(a["dx_px"],
                                                 a["dy_px"])), 1)}


def migrate_success(table, fields):
    """Rename success -> task_success, add pick_success from the stage."""
    if "success" in fields:
        fields[fields.index("success")] = "task_success"
        for r in table:
            r["task_success"] = r.pop("success")
    if "pick_success" not in fields:
        fields.insert(fields.index("task_success") + 1, "pick_success")
        for r in table:
            st = r.get("stage_reached", "")
            r["pick_success"] = int(st in PICKED) if st else ""


def update_results(model, spot, rows):
    if not RESULTS.exists():
        return
    with open(RESULTS) as f:
        table = list(csv.DictReader(f))
        fields = list(table[0].keys()) if table else []
    migrate_success(table, fields)
    new_cols = [("recovery", "miss_direction")]
    new_cols += [(c, "notes") for c in CAM_FIELDS]
    new_cols += [(c, "notes") for c in TIME_FIELDS]
    new_cols += [(c, "notes") for c in DISTRACTOR_FIELDS]
    for c, before in new_cols:
        if c not in fields:
            fields.insert(fields.index(before), c)
            for r in table:
                r.setdefault(c, "")
    mine = [r for r in table if r["model"] == model and r["spot"] == spot]
    by_src = {src_tag(r["notes"]): r for r in mine if src_tag(r["notes"])}
    filled = updated = 0
    queue = []
    for k in sorted(rows):
        old = by_src.get(src_tag(rows[k]["notes"]))
        if old is not None:
            old.update(rows[k])
            updated += 1
        else:
            queue.append(rows[k])
    for r in mine:
        if r["stage_reached"]:
            continue
        if not queue:
            break
        r.update(queue.pop(0))
        filled += 1
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    print(f"filled {filled}, updated {updated} rows in {RESULTS.name} "
          f"({model} {spot})")
    if queue:
        print(f"WARNING: {len(queue)} episodes had no empty row left")


def src_tag(notes):
    for part in notes.split("; "):
        if part.startswith("src="):
            return part
    return ""


def parse_rollout_name(name):
    """Extract (model, spot) from rollout_<model>_<spot>_<YYYYMMDD>_<HHMMSS>."""
    if not name.startswith("rollout_"):
        return "", ""
    rest = name[len("rollout_"):]
    for m in sorted(CHECKPOINTS, key=len, reverse=True):
        prefix = m + "_"
        if rest.startswith(prefix):
            spot = rest[len(prefix):].split("_")[0]
            return m, spot
    return "", ""


def main():
    name = newest(sys.argv[1])
    cal = calibration()
    ds, s, a, e = load(name)
    model, spot = parse_rollout_name(name)
    OUT.mkdir(exist_ok=True)
    cam = camera_columns(name)
    sheet_rows, auto_rows = [], {}
    for ep in np.unique(e):
        sel = np.flatnonzero(e == ep)
        ev, stage, notes, direction, cm, recovery = score(
            ds, s, a, sel, cal, spot)
        close, hold, rel = ev["_idx"]
        hz, place_s, jerk = timing(a[sel], rel, stage)
        dis = (distractor(ds, s, a, sel, cal)
               if spot.endswith(("D", "DUO")) else {})
        auto_rows[int(ep) + 1] = {
            **dis,
            "stage_reached": stage,
            "task_success": int(stage == "place"),
            "pick_success": int(stage in PICKED),
            "recovery": recovery,
            "miss_direction": direction, "miss_cm": cm,
            **cam,
            "loop_hz": hz, "place_s": place_s, "jerk": jerk,
            "notes": f"{notes}; src={name}:ep{ep}; auto"}
        print(f"trial {ep + 1}: stage={stage} task="
              f"{int(stage == 'place')} pick={int(stage in PICKED)} "
              f"recovery={recovery} "
              f"miss={direction} {cm}cm | {hz}Hz place={place_s}s "
              f"jerk={jerk} | {notes}"
              + (f" | blue {dis['blue_train_cm']}cm from train, "
                 f"{dis['blue_green_cm']}cm from green, close "
                 f"frac={dis['close_frac']} off={dis['close_off_cm']}cm"
                 if dis else ""))
        lift = None if hold is None else min(hold + FPS, len(sel) - 1)
        moments = [("start", 0), ("first close", close),
                   ("hold +1s", lift), ("release", rel),
                   ("end", len(sel) - 1)]
        tiles = [blank(f"ep{ep} {lab}: none") if k is None
                 else tile(ds, sel[k], f"ep{ep} {lab} t={k / FPS:.1f}s")
                 for lab, k in moments]
        sheet_rows.append(np.hstack(tiles))
    sheet = OUT / f"{name}.jpg"
    cv2.imwrite(str(sheet), np.vstack(sheet_rows),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    (OUT / f"{name}.json").write_text(json.dumps(auto_rows, indent=1))
    print(f"saved {sheet}")
    if model in CHECKPOINTS:
        update_results(model, spot, auto_rows)


if __name__ == "__main__":
    main()
