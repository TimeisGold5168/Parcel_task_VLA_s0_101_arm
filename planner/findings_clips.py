"""Clips for the skill-text findings (portfolio page).

    python findings_clips.py

Same format as the other page clips: top camera 636x360 (or its tray
area, 504x360) next to the side or wrist camera 480x360, 30 fps,
H.264, real time. Writes to site/videos:
  skill_switch_top_bag.mp4  Test C ep 0: top-box text (start text and
                            key 3), then key 2 (bag text). The live
                            skill and its task text are drawn on the
                            clip
  grasp_gap.mp4             drive run ep 0: the gripper closes in the
                            gap between the black box and the black bag

The key log (debug_logs/skill_20261119_191600.csv) has wall-clock
times only. Key 2 is placed at 35.0 s of the episode: the arm is still
at home from 29.5 s and starts to move at 35.5 s, and key 3 (29.4 s
earlier) then falls in the first top-box approach.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
SITE = Path(__file__).resolve().parents[1] / "site/videos"
TRAY = "crop=448:320:0:0,scale=504:360"
KEY2_S = 35.0
TEXTS = {"top": "lift upper carton off stacked pile",
         "bag": "pinch soft bag by its edge then drop onto table"}
# name: (dataset, episode, second camera, start s, end s, top camera
#        filter, skill overlays [(skill, source, from s, to s)])
CLIPS = {
    "skill_switch_top_bag": (
        "rollout_bench_parcel_smolvla_b_sk3d_skilltext_c", 0, "side",
        0.0, 46.0, "scale=636:360",
        [("top", "at start", 0.0, KEY2_S),
         ("bag", "human input", KEY2_S, 46.0)]),
    "grasp_gap": (
        "rollout_bench_parcel_smolvla_b_sk3d_planner_drive", 0, "wrist",
        6.0, 14.0, TRAY, []),
}


def overlay(skills, t0):
    """drawtext filters: the live skill and its task text, bottom left."""
    out = []
    box = "box=1:boxcolor=black@0.6:boxborderw=8"
    for skill, src, a, b in skills:
        on = f"enable='between(t,{a - t0},{b - t0})'"
        out.append(f"drawtext=text='skill\\: {skill}  ({src})':x=12:"
                   f"y=h-80:fontsize=26:fontcolor=0xffc800:{box}:{on}")
        out.append(f"drawtext=text='task text\\: {TEXTS[skill]}':x=12:"
                   f"y=h-38:fontsize=18:fontcolor=white:{box}:{on}")
    return "".join("," + f for f in out)


def source(root, row, cam):
    k = f"videos/observation.images.{cam}/"
    path = (root / k / f"chunk-{int(row[k + 'chunk_index']):03d}"
            / f"file-{int(row[k + 'file_index']):03d}.mp4")
    return path, float(row[k + "from_timestamp"])


def make(name, ds, ep, cam2, t0, t1, top, texts):
    root = LOW / ds
    eps = pd.concat(pd.read_parquet(f) for f in
                    sorted((root / "meta/episodes").rglob("*.parquet")))
    row = eps.set_index("episode_index").loc[ep]
    inputs = []
    for cam in ("top", cam2):
        path, start = source(root, row, cam)
        inputs += ["-ss", f"{start + t0:.3f}", "-t", f"{t1 - t0:.3f}",
                   "-i", str(path)]
    graph = (f"[0:v]{top}[a];[1:v]scale=480:360[b];"
             f"[a][b]hstack=inputs=2,fps=30{overlay(texts, t0)}[v]")
    out = SITE / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs,
                    "-filter_complex", graph, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-crf", "26", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out)], check=True)
    poster = SITE / f"{name}.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss",
                    f"{(t1 - t0) / 2:.2f}", "-i", str(out),
                    "-frames:v", "1", "-q:v", "4", str(poster)],
                   check=True)
    print(out)


def main(names):
    for name in names or CLIPS:
        make(name, *CLIPS[name])


if __name__ == "__main__":
    main(sys.argv[1:])
