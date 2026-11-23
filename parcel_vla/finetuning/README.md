# finetuning (phase 04)

This folder holds the DAgger corrections and the fine-tune rounds of
SmolVLA B. When the robot fails, the operator takes over with the
leader arm and records the fix.

| File | Purpose |
|---|---|
| `dagger_parcel.sh` | Record corrections: the policy drives, Tab starts and saves a correction |
| `trim_corrections.py` | Cut the corrections to the pick and the place, without the idle tail at home |
| `start_test.py` | Offline start test: does the model still leave the home pose? |
| `pick_checkpoint.py` | Score the checkpoints of a fine-tune and pick one |
| `early_check.py` | Early offline check of an all-skill fine-tune |
| `queue_dagger.sh` | Round 1: merge all corrections into dataset B, then fine-tune |
| `queue_dagger_gentle.sh` | Round 1 again with a lower LR (gentle fine-tune) |
| `queue_control_b.sh` | Control run: the same recipe on dataset B only |
| `queue_skill.sh` | Rounds 2 and 3: one model for each new skill (far edge, box stack) |
| `queue_bsp_skill.sh` | SmolVLA-BSP version of a new-skill model |

The `queue_*.sh` scripts record the exact recipe of each round. Each
one waits for a free GPU, then runs its steps in order.

## Results

| Round | Data | Result |
|---|---|---|
| 1. All failures at once | 83 corrections, 45 % of the frames | The arm stays at home (start roll 32° → 3.7°) |
| 2. Far edge | 16 short corrections, 7 % of the frames | Error 16.3° → 2.4°; works on the robot |
| 3. Stacked boxes | 20 full demos + 30 short corrections | Error 3.5° → 1.8°; 3 of 5 stacks |

The control run (dataset B only, same recipe) kept 27° of start roll.
Thus the data caused the round 1 failure, not the recipe.

## Lessons

1. Add one skill at a time. Use short corrections with only the pick
   and the place, at a small share of the frames.
2. Before a robot run, check the start motion offline
   (`start_test.py`). A failed checkpoint stops here.
3. A correction fixes a skill that the model already has. A new skill
   (for example, a box on a box: 5 of 99 demos) needs about 20 good
   full demos.

```bash
# record 10 corrections with the benchmark model (step 10000)
bash parcel_vla/finetuning/dagger_parcel.sh smolvla_b_fb 10 far_edge
# start test of two checkpoints
python -m parcel_vla.finetuning.start_test smolvla_b smolvla_b_fb:010000
```

The skill reports of rounds 2-4 are in
[`results/finetuning/`](../../results/finetuning).
