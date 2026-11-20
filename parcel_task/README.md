# parcel_task

This folder holds the tools of the parcel task: recording, training,
offline tests and robot runs. Run all commands from the LeRobot root.
The results and the method are in [`../writeup.md`](../writeup.md).

| File | Purpose |
|---|---|
| `record_parcel.sh` | Record one heap of a schedule with 3 cameras |
| `record_preview.py` | `lerobot-record` with a live 3-camera preview |
| `check_episodes.py` | Check the episode lengths, marks and frames |
| `make_schedule.py`, `schedule_*.csv` | Heap schedules |
| `train_mixed.py` | Diffusion training with mixed camera sizes |
| `queue_b.sh` | Train the four dataset B models in a row |
| `queue_skill.sh`, `queue_all.sh` | New-skill and all-skill fine-tunes |
| `make_skill3.py`, `queue_skill3.sh` | Skill-text datasets and models |
| `offline_eval.py`, `offline_eval_b.py` | Offline tests (no robot) |
| `start_test.py`, `pick_checkpoint.py` | Start test and checkpoint pick |
| `lang_test.py` | Language test of the skill-text models |
| `eval_parcel.sh` | One entry point for all robot tests |
| `run_parcel.sh`, `rollout_mixed.py` | Robot runs through `lerobot-rollout` |
| `run_bsp_parcel.py` | Robot runs of the BSP models |
| `dagger_parcel.sh`, `trim_corrections.py` | DAgger corrections |
| `bench_sheets.py` | Frame sheets and a score template |

Subfolders: `figures/` (figures of the write-up), `offline_eval/` and
`offline_eval_b/` (offline results), `eval_logs/` and `debug_logs/`
(robot run logs).
