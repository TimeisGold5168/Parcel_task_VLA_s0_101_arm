# planner (phase 05)

This folder holds the skill planner. At the home pose, the planner
looks at the camera images and selects the skill text for SmolVLA. On
the way to the parcel, the wrist camera checks the skill. The tests
and results are in [`writeup.md`](../../docs/writeup.md), sections
"Planner test" and "Findings: skill text and planner".

| File | Purpose |
|---|---|
| `collect_home_frames.py` | Collect home-pose frames for the test set |
| `labels.csv`, `frames/index.csv` | Labels and index of the test set (the frames are local) |
| `vlm_planner.py` | Zero-shot planner: yes/no questions to a VLM, priority in code |
| `stack_variants.py` | Camera and prompt variants for the stack question |
| `probe.py` | Linear probe on the VLM features |
| `next_kind.py` | Probe: kind of the next parcel, from home |
| `wrist_kind.py` | Probe: kind of the parcel in the wrist camera |
| `train_planner.py` | Train the home and wrist probes (`planner_500m.npz`) |
| `topview.py` | Colour regions in the top view, ranked |
| `live_planner.py` | Planner thread next to the policy on the robot |

New result tables go to `runs/planner/`. My results are in
[`results/planner/`](../../results/planner).

## Results

| Method | Skill accuracy | Time |
|---|---|---|
| Zero-shot SmolVLM2-2.2B, top + side | 69 % | 1.5 s |
| Linear probe on SmolVLM2-2.2B, 3 cameras | 81 % | about 2 s |
| Next-parcel probe, SmolVLM2-500M, home | 74 % | 0.34 s |
| Wrist probe, SmolVLM2-500M, 1 s before the grasp | 99 % | 0.09 s |

SmolVLM2-2.2B (about 4.5 GB) does not fit next to SmolVLA on a 6 GB
GPU. SmolVLM2-500M fits (1.9 GB at peak). On the robot, the home probe
did not transfer to new heaps, and a black box that touches a black
bag looks like one parcel. The live planner is not reliable yet.

```bash
python -m parcel_vla.planner.vlm_planner --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
python -m parcel_vla.planner.train_planner
SKILL=bag PLANNER=drive bash parcel_vla/robot/run_parcel.sh smolvla_b_sk3d
```
