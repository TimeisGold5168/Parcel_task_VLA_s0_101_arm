# evaluation (phase 03)

This folder holds the offline tests. They use held-out demos, not the
robot. Every model change passed an offline test before a robot run.

| File | Purpose |
|---|---|
| `offline_eval_b.py` | Tests 1-3 of the dataset B models |
| `offline_eval.py` | Tests of the dataset A1 models; the model list and loaders that other scripts use |
| `box_fit_test.py` | Does a model plan the grasp pose of its own demos? |
| `inspect_heap.py` | One recorded heap: videos, marks, schedule and demo quality |
| `home_scan.py` | Episodes that start or end away from the home pose |

## The three tests

| Test | Question | Better |
|---|---|---|
| 1. Action error | Mean error of the 5 arm joints over the next 1 s (deg) | lower |
| 2. Flip decision | Does the model flip the parcels with the label up? (99 episodes) | higher |
| 3. Image swap | Do the predictions follow swapped camera images? | higher |

```bash
python -m parcel_vla.evaluation.offline_eval_b
python -m parcel_vla.evaluation.offline_eval_b --models smolvla_b diffusion_b
python -m parcel_vla.evaluation.home_scan ~/.cache/huggingface/lerobot/Low101/parcel-sort-b
```

The results go to `runs/offline_eval_b/`. My results are in
[`results/offline_eval_b/`](../../results/offline_eval_b).

Lesson: the offline ranking is not the robot ranking. SmolVLA had the
lowest error (1.91°), but Diffusion-BSP needed the least help on the
robot.
