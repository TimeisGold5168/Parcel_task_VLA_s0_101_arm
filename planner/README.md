# planner

This folder holds the skill planner. At the home pose, the planner
looks at the camera images and selects the skill text for SmolVLA.
The tests and results are in [`../writeup.md`](../writeup.md),
sections "Planner test" and "Findings: skill text and planner".

| File | Purpose |
|---|---|
| `collect_home_frames.py` | Collect home-pose frames for the test set |
| `labels.csv`, `frames/` | Labels and frames of the test set |
| `vlm_planner.py` | Zero-shot planner: yes/no questions to a VLM |
| `stack_variants.py` | Camera and prompt variants for the stack question |
| `probe.py` | Linear probe on the VLM features |
| `next_kind.py` | Probe: kind of the next parcel, from home |
| `wrist_kind.py` | Probe: kind of the parcel in the wrist camera |
| `train_planner.py` | Train the home and wrist probes |
| `topview.py` | Colour regions in the top view, ranked |
| `live_planner.py` | Planner thread next to the policy on the robot |
| `findings_figure.py` | Make `parcel_task/figures/skill_text_findings.jpg` |
| `findings_clips.py` | Make the findings clips for the web page |

The `*.csv` and `*.log` files are the test results.
