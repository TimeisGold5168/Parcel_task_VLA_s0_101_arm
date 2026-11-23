"""Live box/bag planner on the robot (SmolVLM2-500M + 2 probes).

Runs in a thread next to the policy (rollout_mixed.py, PLANNER=log or
drive). It reads the robot observations and calls:
  plan    once per home visit, after HOME_WAIT s in the rest pose:
          probe  the home probe of train_planner.py (top + side +
                 wrist crops): the kind of the parcel that the operator
                 took next in dataset B (the habit of the policy)
          view   topview.py: the parcels ranked by the row nearest to
                 the robot, then nearest to the flip zone
          Both are logged; PLAN_FROM (probe or view, default probe)
          gives the skill. No stack check: the top-box skill comes
          from the operator (key 3).
  verify  every WRIST_EVERY s for WRIST_FOR s after the arm leaves
          home, until the grasp (the gripper closes 10 below its
          highest value after it opens above 25): box or bag in front
          of the gripper (wrist probe), and if it agrees with the text
          in use
log:   only logs the calls; the operator keeps the keys.
drive: the plan call sets the text. With WRIST_SWITCH=1, the first
       verify call with P >= SURE that does not agree changes it one
       time (off by default: two parcels in the wrist image can give a
       wrong call).
Each call goes to the CSV log with its image in <log>_img/.
"""

import csv
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from parcel_vla.planner import topview
from parcel_vla.planner.probe import WRIST_CROP, feature
from parcel_vla.planner.vlm_planner import SIDE_CROP, TOP_CROP, VLMPlanner, crop

HOME_WAIT = 0.5  # s in the rest pose before the arm counts as home
AWAY_WAIT = 0.3  # s out of it before the arm counts as away
WRIST_EVERY, WRIST_FOR = 1.0, 6.0
SURE = 0.8  # WRIST_SWITCH: P(skill) needed to change the text
COLUMNS = ["time", "call", "skill", "p_bag", "text_in_use", "agree",
           "gripper", "ms", "plan", "image"]


def rest_pose(obs):
    """Rest pose of bench_sheets.home_time (shoulder lift, elbow)."""
    return obs["shoulder_lift.pos"] < -95 and obs["elbow_flex.pos"] > 85


class Probes:
    def __init__(self, path):
        z = np.load(path)
        self.vlm = VLMPlanner(str(z["model"]))
        self.question = {k: str(z[f"{k}_question"]) for k in ("home",
                                                               "wrist")}
        self.lin = {k: (z[f"{k}_mu"], z[f"{k}_sd"], z[f"{k}_w"],
                        float(z[f"{k}_b"])) for k in ("home", "wrist")}

    def p_bag(self, kind, images):
        f = feature(self.vlm, images, self.question[kind])
        mu, sd, w, b = self.lin[kind]
        return float(1 / (1 + np.exp(-(((f - mu) / sd) @ w + b))))


class LivePlanner(threading.Thread):
    def __init__(self, probes_path, log_path, drive, set_skill, skill_now):
        super().__init__(daemon=True)
        self.probes_path, self.drive = probes_path, drive
        self.set_skill, self.skill_now = set_skill, skill_now
        self.log_path = Path(log_path)
        self.img_dir = self.log_path.with_name(self.log_path.stem + "_img")
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.obs = None

    def update(self, obs):
        with self.lock:
            self.obs = (time.time(), obs)

    def write(self, **row):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])

    def plan(self, t, obs, grip):
        t0 = time.perf_counter()
        rgb = {c: Image.fromarray(obs[c]) for c in ("top", "side",
                                                    "wrist")}
        p = self.probes.p_bag("home", [crop(rgb["top"], TOP_CROP),
                                       crop(rgb["side"], SIDE_CROP),
                                       crop(rgb["wrist"], WRIST_CROP)])
        ranked = topview.parcels(obs["top"])
        ms = (time.perf_counter() - t0) * 1000
        by_probe = "bag" if p >= 0.5 else "box"
        by_top = ranked[0][0] if ranked else ""
        skill = by_top if self.plan_from in ("view", "top") else by_probe
        name = f"{t:.3f}_plan.jpg"
        cv2.imwrite(str(self.img_dir / name),
                    topview.draw(obs["top"], ranked))
        plan = " ".join(f"{k}({c})" for k, c, _, _, _ in ranked)
        self.write(time=f"{t:.3f}", call="plan", skill=skill,
                   p_bag=f"{p:.3f}", text_in_use=self.skill_now(),
                   gripper=f"{grip:.1f}", ms=f"{ms:.0f}",
                   plan=f"probe {by_probe}; view {plan}", image=name)
        print(f"[planner] plan: probe {by_probe} (P bag {p:.2f}), view "
              f"{plan or 'tray empty'} -> {skill or '-'} "
              f"(from {self.plan_from}; no stack check)", flush=True)
        return skill

    def verify(self, t, wrist, grip):
        t0 = time.perf_counter()
        p = self.probes.p_bag("wrist", [Image.fromarray(wrist)])
        ms = (time.perf_counter() - t0) * 1000
        skill = "bag" if p >= 0.5 else "box"
        agree = skill == self.skill_now()
        name = f"{t:.3f}_verify.jpg"
        cv2.imwrite(str(self.img_dir / name),
                    cv2.cvtColor(wrist[::2, ::2], cv2.COLOR_RGB2BGR))
        self.write(time=f"{t:.3f}", call="verify", skill=skill,
                   p_bag=f"{p:.3f}", text_in_use=self.skill_now(),
                   agree=int(agree), gripper=f"{grip:.1f}",
                   ms=f"{ms:.0f}", image=name)
        print(f"[planner] verify: {skill} (P bag {p:.2f}), text in use "
              f"{self.skill_now()}: {'agree' if agree else 'DISAGREE'}",
              flush=True)
        return skill, max(p, 1 - p), agree

    def run(self):
        self.probes = Probes(self.probes_path)
        self.plan_from = os.environ.get("PLAN_FROM", "probe")
        wrist_switch = os.environ.get("WRIST_SWITCH", "0") != "0"
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(COLUMNS)
        print(f"[planner] ready ({'drive' if self.drive else 'log only'}, "
              f"plan from {self.plan_from}, wrist switch "
              f"{'on' if wrist_switch else 'off'}) -> {self.log_path}",
              flush=True)
        last_t = rest_since = away_since = left_at = grip_max = None
        home = switched = False
        last_wrist = 0.0
        while True:
            time.sleep(0.1)
            with self.lock:
                snap = self.obs
            if snap is None or snap[0] == last_t:
                continue
            t, obs = snap
            last_t = t
            if rest_pose(obs):
                rest_since, away_since = rest_since or t, None
            else:
                rest_since, away_since = None, away_since or t
            grip = obs["gripper.pos"]
            if not home and rest_since and t - rest_since >= HOME_WAIT:
                home, left_at = True, None
                with torch.no_grad():
                    skill = self.plan(t, obs, grip)
                if self.drive and skill:
                    self.set_skill(skill)
            if home and away_since and t - away_since >= AWAY_WAIT:
                home, left_at, grip_max = False, away_since, None
                switched = False
            if home or not left_at:
                continue
            if grip > 25:
                grip_max = max(grip_max or grip, grip)
            if grip_max and grip < grip_max - 10:
                self.write(time=f"{t:.3f}", call="grasp",
                           text_in_use=self.skill_now(),
                           gripper=f"{grip:.1f}")
                print("[planner] grasp", flush=True)
                left_at = None
            elif t - left_at <= WRIST_FOR and t - last_wrist >= WRIST_EVERY:
                last_wrist = t
                with torch.no_grad():
                    skill, sure, agree = self.verify(t, obs["wrist"], grip)
                if self.drive and wrist_switch and not switched \
                        and sure >= SURE and not agree:
                    self.set_skill(skill)
                    switched = True
