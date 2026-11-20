"""Cut the source footage for the portfolio video, for review.

Each clip plays at 1x. Dataset clips show the top, wrist and side
cameras next to each other. A bar on top gives the source and the time
from the start of the episode.

Run: python outputs/portfolio/video/make_footage.py [name ...]
Output: outputs/portfolio/video/footage/
"""

import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
PORT = Path(__file__).resolve().parents[1]
RUNS = PORT / "bsp/runs"
SITE = PORT / "site"
OUT = Path(__file__).resolve().parent / "footage"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CAMS = ["top", "wrist", "side"]
W, H, BAR = 480, 360, 40


def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")


def bar_filter(label, t0):
    """Black bar on top with the label and the episode time."""
    t = f"(t+{t0:.2f})"
    timer = f"%{{eif\\:{t}\\:d}}.%{{eif\\:mod({t}*10\\,10)\\:d}} s"
    return (f"pad=iw:ih+{BAR}:0:{BAR}:black,"
            f"drawtext=fontfile={FONT}:text='{esc(label)}':x=12:y=11:"
            f"fontsize=18:fontcolor=white,"
            f"drawtext=fontfile={FONT}:text='{timer}':x=w-tw-12:y=11:"
            f"fontsize=18:fontcolor=yellow")


def encode(cmd, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cmd, "-r", "30",
                    "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    print("wrote", out.name)


def episodes(ds):
    files = glob.glob(str(LOW / ds / "meta/episodes/*/*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in files])
    return eps.set_index("episode_index")


TEXTS = {"bag": "pinch soft bag by its edge then drop onto table",
         "box": "grasp rigid box with wide grip and set it at flip zone",
         "top": "lift upper carton off stacked pile"}


def drive_events(ep):
    """(episode time, skill, source) from footage/skill_timeline.csv."""
    rows = [r for r in csv.DictReader(open(OUT / "skill_timeline.csv"))
            if int(r["episode"]) == ep]
    return [(float(r["t_s"]), r["skill"], r["source"]) for r in rows]


def skill_filter(events, start, dur):
    """Live skill text and task text for each event (clip time)."""
    ev = list(events) + [(start + dur + 1, "", "")]
    parts = []
    for (t0, sk, src), (t1, _, _) in zip(ev, ev[1:]):
        a, b = max(t0 - start, 0), min(t1 - start, dur)
        if b <= 0 or a >= dur or not sk:
            continue
        on = f"enable='between(t,{a:.2f},{b:.2f})'"
        box = "box=1:boxcolor=black@0.55:boxborderw=8"
        parts.append(
            f"drawtext=fontfile={FONT}:text='skill\\: {sk}  ({src})':"
            f"x=16:y=h-86:fontsize=30:fontcolor=0xffc800:{box}:{on}")
        parts.append(
            f"drawtext=fontfile={FONT}:text='task text\\: {TEXTS[sk]}':"
            f"x=16:y=h-38:fontsize=20:fontcolor=white:{box}:{on}")
    return ",".join(parts)


def cam_inputs(ds, ep, start, dur):
    """ffmpeg inputs and scale chains for the 3 cameras of an episode."""
    r = episodes(ds).loc[ep]
    ins, chains = [], []
    for i, cam in enumerate(CAMS):
        k = f"videos/observation.images.{cam}"
        f = (LOW / ds / "videos" / f"observation.images.{cam}" /
             f"chunk-{int(r[k + '/chunk_index']):03d}" /
             f"file-{int(r[k + '/file_index']):03d}.mp4")
        t = r[k + "/from_timestamp"] + start
        ins += ["-ss", f"{t:.3f}", "-t", f"{dur:.3f}", "-i", str(f)]
        chains.append(f"[{i}:v]scale={W}:{H},setpts=PTS-STARTPTS[v{i}]")
    return ins, ";".join(chains) + ";[v0][v1][v2]hstack=3"


def ds_clip(out, ds, ep, start=0.0, dur=None, label="", skills=None):
    """3 cameras of one dataset episode, from start for dur seconds.

    skills: list of (episode time, skill, source) for the skill text.
    """
    length = episodes(ds).loc[ep]["length"] / 30
    dur = min(dur or length, length - start)
    ins, fc = cam_inputs(ds, ep, start, dur)
    lab = label or f"{ds}  ep{ep}"
    fc += "," + bar_filter(f"{lab}  (top | wrist | side, 1x)", start)
    if skills:
        fc += "," + skill_filter(skills, start, dur)
    encode([*ins, "-filter_complex", fc], OUT / out)


def ds_still(out, ds, ep, t, label):
    """One frame of the 3 cameras at episode time t, as a JPEG."""
    ins, fc = cam_inputs(ds, ep, t, 0.5)
    fc += "," + bar_filter(f"{label}  (top | wrist | side)", t)
    (OUT / out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex",
                    fc, "-frames:v", "1", "-q:v", "2", str(OUT / out)],
                   check=True)
    print("wrote", out)


def run_clip(out, run, eps, label):
    """BSP run episodes (ep<n>_cams.mp4) one after the other."""
    parts = []
    for ep in eps:
        f = RUNS / run / f"ep{ep}_cams.mp4"
        if not f.exists():
            f = RUNS / run / f"ep{ep}_front.mp4"
        tmp = OUT / f".tmp_{run}_ep{ep}.mp4"
        encode(["-i", str(f), "-vf",
                "scale=1440:-2," + bar_filter(f"{label}  ep{ep}  (1x)", 0)],
               tmp)
        parts.append(tmp)
    lst = OUT / ".concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    encode(["-f", "concat", "-safe", "0", "-i", str(lst)], OUT / out)
    for p in parts:
        p.unlink()
    lst.unlink()


def copy(out, src):
    (OUT / out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUT / out)
    print("copied", out)


DRIVE = "rollout_bench_parcel_smolvla_b_sk3d_planner_drive"
TEST_C = "rollout_bench_parcel_smolvla_b_sk3d_skilltext_c"
# Test C, episode 0: launch skill top; key 2 (bag) at about 35 s
TEST_C_EP0 = [(0.0, "top", "at start"), (35.0, "bag", "human input")]

CLIPS = {
    # Scene 1: latest model (sk3d). Episode 3: planner changes only.
    **{f"s1_hook_sk3d_drive_ep{e}": (lambda e=e: ds_clip(
        f"s1_hook/sk3d_planner_drive_ep{e}_full.mp4", DRIVE, e,
        label=f"SmolVLA sk3d, planner drive, ep{e}",
        skills=drive_events(e)))
       for e in (3, 4, 7)},
    # Scene 2: push approach.
    "s2_push_demo": lambda: ds_clip(
        "s2_push/a1_demo_ep14_black_bag.mp4", "parcel-sort-a", 14,
        label="Dataset A1 demo ep14, black bag (push)"),
    "s2_push_diffusion_bsp": lambda: run_clip(
        "s2_push/push_model_diffusion_bsp_a1_ep0-1.mp4",
        "diffusion_bsp_parcel_easy_20261103_190342", [0, 1],
        "Push model: Diffusion-BSP (A1), 1.5x speed setting"),
    "s2_push_diffusion_retry": lambda: ds_clip(
        "s2_push/push_model_diffusion_a1_retries.mp4",
        "rollout_eval_parcel_diffusion_easy", 0,
        label="Push model: Diffusion (A1), push retries"),
    "s2_push_smolvla": lambda: ds_clip(
        "s2_push/push_model_smolvla_up.mp4",
        "rollout_parcel_smolvla_up", 0, label="Push model: SmolVLA (A1)"),
    # Scene 3: box pushes from the far edge; then pick and place.
    **{f"s3_push_box_ep{e}": (lambda e=e: ds_clip(
        f"s3_why_pick/a1_demo_ep{e}_box_far_edge.mp4", "parcel-sort-a",
        e, label=f"Dataset A1 demo ep{e}, box at the far edge (push)"))
       for e in (6, 48, 69, 75, 99)},
    "s3_pick_demo": lambda: ds_clip(
        "s3_why_pick/b_demo_ep2_box_pick_place.mp4", "parcel-sort-b", 2,
        label="Dataset B demo ep2, box (pick and place)"),
    # Scene 4: same dataset B, both at 1.5x, both work.
    "s4_diffusion_bsp": lambda: run_clip(
        "s4_compare/diffusion_bsp_heap5_ep0-2.mp4",
        "diffusion_bsp_b_parcel_hard_20261111_184111", [0, 1, 2],
        "Diffusion-BSP B, 5-parcel heap, 1.5x speed setting"),
    "s4_smolvla_bsp": lambda: run_clip(
        "s4_compare/smolvla_bsp_heap5_ep0-3-4.mp4",
        "smolvla_bsp_b_parcel_hard_20261111_183512", [0, 3, 4],
        "SmolVLA-BSP B, 5-parcel heap, 1.5x speed setting"),
    # Scene 5: far edge before, correction, after; box on box.
    "s5_far_edge_before": lambda: run_clip(
        "s5_corrections/far_edge_before_diffusion_bsp_heap8_ep2.mp4",
        "diffusion_bsp_b_parcel_hard_20261111_190455", [2],
        "Diffusion-BSP B, 8-parcel heap, far edge"),
    **{f"s5_takeover_ep{e}": (lambda e=e: ds_clip(
        f"s5_corrections/far_edge_takeover_ep{e}.mp4",
        "rollout_dagger_far_edge", e,
        label=f"Far-edge correction ep{e} (leader arm takeover)"))
       for e in (5, 15, 7)},
    "s5_far_edge_after": lambda: ds_clip(
        "s5_corrections/far_edge_after_smolvla_fe.mp4",
        "rollout_eval_parcel_smolvla_b_fe_hard", 0,
        label="SmolVLA + far-edge corrections (fe)"),
    "s5_box_on_box": lambda: copy(
        "s5_corrections/box_on_box_robot_fbs.mp4",
        SITE / "videos/fbs_box_on_box.mp4"),
    "s5_box_demo": lambda: copy(
        "s5_corrections/box_on_box_demo.mp4",
        SITE / "videos/demo_box_stack.mp4"),
    # Scene 6: background for the benchmark bars.
    "s6_bench_bg": lambda: ds_clip(
        "s6_benchmark/bench_kind1_ep0_orange_boxes.mp4",
        "rollout_bench_parcel_smolvla_b_fb_kind1_many", 0, 0, 60,
        label="Benchmark, 4 orange boxes, no help"),
    # Scene 8: top-box skill, then the bag skill by human input.
    "s8_skill_switch": lambda: ds_clip(
        "s8_skill_texts/skill_switch_ep0.mp4", TEST_C, 0,
        label="SmolVLA sk3d, skill text test C, ep0",
        skills=TEST_C_EP0),
    # Scene 9: limits.
    "s9_grasp_gap_ep2": lambda: ds_still(
        "s9_limits/grasp_gap_drive_ep2_15.9s.jpg", DRIVE, 2, 15.9,
        "Gripper closes between black box and black bag, ep2"),
    "s9_merged_region": lambda: copy(
        "s9_limits/merged_region_figure.jpg",
        SITE / "assets/skill_text_findings.jpg"),
    "s9_home_wait": lambda: ds_clip(
        "s9_limits/home_wait_kind1_ep3.mp4",
        "rollout_bench_parcel_smolvla_b_fb_kind1_many", 3, 15, 36,
        label="Benchmark, black boxes: still at home from 20 s"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or CLIPS:
        CLIPS[name]()
