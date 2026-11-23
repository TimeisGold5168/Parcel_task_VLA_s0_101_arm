# robot (phases 03-06)

This folder runs the trained policies on the robot.

| File | Purpose |
|---|---|
| `run_parcel.sh` | Run a Diffusion or SmolVLA model through `lerobot-rollout` |
| `rollout_mixed.py` | `lerobot-rollout` with the parcel fixes: GPU image prepare, preview, RTC fixes, skill keys, live planner |
| `run_bsp_parcel.py` | B-spline executor for the parcel task (Diffusion-BSP, SmolVLA-BSP) |
| `run_bsp.py` | B-spline executor of the cube task, and the base class of `run_bsp_parcel.py` |
| `video_record.py` | Full-rate video recording of a run |
| `rollout_debug.py` | Compare the debug logs of `run_parcel.sh DEBUG=1` |

## Before a run

CAUTION: The policy moves the arm. Clear the workspace before you
start.

1. Set your devices (see [`../env.sh`](../env.sh)).
2. Put the arm at the home pose. The robot goes back to its start
   pose after each episode.
3. Build the heap.

## Commands

```bash
# SmolVLA with skill texts: keys 1/2/3 send the box, bag or top-box text
SKILL=bag bash parcel_vla/robot/run_parcel.sh smolvla_b_sk3d

# the live planner sends the skill (PLANNER=log: it only logs its calls)
SKILL=bag PLANNER=drive bash parcel_vla/robot/run_parcel.sh smolvla_b_sk3d

# SmolVLA with RTC and the three fixes
FREEZE=1 RTC_EXACT_DELAY=1 COMPILE=default ROLLOUT_ARGS="--inference.type=rtc \
  --inference.rtc.execution_horizon=10 --inference.queue_threshold=30" \
  bash parcel_vla/robot/run_parcel.sh smolvla_b

# Diffusion-BSP at 1.5x with the smooth settings
python -m parcel_vla.robot.run_bsp_parcel --model diffusion_bsp_b --level easy \
  --speed 1.5 --ddim-steps 5 --blend 0.2 --max-step 6

# a checkpoint folder of your own (or from the Hub)
POLICY_PATH=/path/to/pretrained_model bash parcel_vla/robot/run_parcel.sh smolvla_b
```

`run_parcel.sh` finds a model by its name in `TRAIN_DIR`, for example
`smolvla_b_sk3d` → `smolvla_parcel_b_sk3d_10k`. The file header lists
all names. `CKPT_STEP=010000` runs another checkpoint of the run.
`RECORD=1` saves the episodes to a dataset.

Keys: Right arrow ends the episode, Left arrow records it again, Esc
stops. Debug logs and B-spline run videos go to `runs/`.

## Settings of the B-spline executor

| Flag | Meaning | Robot runs |
|---|---|---|
| `--speed` | Curve speed against the demos | 1.5 |
| `--blend` | Fade between two segments (s) | 0.2 |
| `--max-step` | Max joint change per 20 ms tick (deg) | 6 |
| `--ddim-steps` | Diffusion denoising steps | 5 |
| `--trt fp16` | TensorRT engines (build them first) | Diffusion |
| `--control cpp` | C++ control process | Diffusion |
| `--policy-path` | Checkpoint folder to run | - |
