# B-spline Policy (BSP) on LeRobot: how each policy is changed

Reference: B-spline Policy, <https://b-spline-policy.github.io/>.
The fit, chunk and decode code in `bspline.py` follows the official
code (MIT licence, `bspline_policy/common/bspline_action.py`).

This document explains how we changed ACT, Diffusion Policy and
SmolVLA so that each one predicts a B-spline segment. It also shows
how the robot runs these segments.

## 1. Why BSP

A standard LeRobot policy predicts an action chunk: one joint target
for each 30 fps frame. The robot runs one action per frame. This has
three problems:

- The speed of the arm is tied to the frame rate of the demos.
- A slow model makes the arm wait for the next chunk.
- Two chunks do not always agree, so the arm jumps at the switch.

A BSP policy predicts one smooth curve (a B-spline segment) in place
of the chunk. The executor samples the curve by the wall clock.
Thus the arm can run the same curve at 1x, 2x or 3x speed, and the
planner can run at a different rate from the control loop.

## 2. The pipeline (the same for all policies)

### 2.1 Targets: `make_targets.py` and `bspline.py`

1. For each episode, fit a cubic B-spline (degree 3) to the 6 joint
   actions. Add knots until the maximum fit error is 0.5 deg or less.
2. Cut the spline into segments of 10 interior knots.
3. For each frame, store one segment as a matrix of 16 rows x 7
   columns:
   - 16 rows = 10 knots + 2 x degree.
   - Column 0 = knot time, in frames, relative to the current frame.
   - Columns 1-6 = control points of the 6 joints (deg).
4. Save all targets in `targets_<dataset>_e0.5.npz`, indexed by the
   global frame index of the dataset.

Parcel dataset `parcel-sort-a` (101,642 frames, 100 episodes):

| Value | Result |
|---|---|
| Fit error | max 0.50 deg, median 0.41 deg |
| Frames per knot | 4.9 (median) |
| Time span of one segment | 42 frames = 1.40 s (median) |

A frame uses the first segment whose first interior knot is at or
after the frame. Thus a segment can start a few frames after "now".
The executor and the offline test handle this (see 2.4 and the
offline evaluation in `../writeup.md`).

### 2.2 Training: `train_bsp.py`

The policy class does not change. Only the action changes:

1. Load the config of the trained baseline checkpoint (`--init`).
2. Set the action feature to shape (7,) and the chunk length to 16
   rows. Each row of the "action chunk" is now one row of the B-spline
   matrix.
3. Replace the action of each dataset item with its B-spline target
   (`BSplineTargets`). No row is padding.
4. Compute the action statistics (mean, std, min, max) on the targets.
   The knot column is normalized like a joint.
5. Warm start: copy every baseline weight that has the same shape. If
   only the first dimension is longer in the baseline, keep the first
   rows (slice). New layers start from random weights.
6. Train with the optimizer and LR schedule presets of the policy.
7. Save a normal LeRobot checkpoint plus `bsp.json` (degree, chunk
   size, max error, fps, rows).

### 2.3 Inference: `segment.py`

`predict_segment(policy, obs)` returns all 16 rows of one segment.
`bspline.decode` or `make_spline` makes a curve from the rows. The
knot column is made non-decreasing before use.

### 2.4 Executor: `run_bsp.py` (cube) and `../parcel_task/run_bsp_parcel.py`

Two threads share the robot (one lock for the motor bus):

- Planner: observation -> one segment. It plans the next segment when
  the current segment is `--replan-frac` used (default 0.5).
- Control (`--hz`, default 50 Hz): samples the active segment at
  tau = elapsed_s x 30 fps x `--speed`.

Other settings:

- A new segment starts at the point near "now" that is closest to the
  last command (as in the official executor).
- `--blend` fades from the old segment to the new segment
  (smoothstep), so the switch does not jump.
- `--max-step` limits each joint step per tick (deg).

## 3. Changes for each policy

| | ACT-BSP | Diffusion-BSP | SmolVLA-BSP |
|---|---|---|---|
| Chunk length | 100 -> 16 (`chunk_size`, `n_action_steps`) | horizon 64 -> 16, `n_action_steps` 32 -> 16 | 50 -> 16 (`chunk_size`, `n_action_steps`) |
| Action width | 6 -> 7 | 6 -> 7 | 7, padded to `max_action_dim` 32 (no change) |
| New layers | `action_head`, `vae_encoder_action_input_proj`, `vae_encoder_pos_enc` | U-Net first conv, its residual conv, `final_conv` | None |
| Sliced layers | `decoder_pos_embed` (100 -> 16 rows) | None | None |
| Weights copied | 229 tensors | 394 (cube), 519 (parcel) | 500 tensors (all) |
| Other config changes | `temporal_ensemble_coeff = None` | `drop_n_last_frames = 0` | `load_vlm_weights = False`, `empty_cameras = 0` |
| Processors | New, with the target stats | New, with the target stats | From `--init` (tokenizer, camera rename); only the action stats change |
| Inference path | `predict_action_chunk` | `conditional_sample` (direct call) | `predict_action_chunk(noise=...)` |

### 3.1 ACT-BSP

- The decoder has one learned query for each chunk row. The baseline
  has 100 queries, and BSP needs 16. We keep the first 16 query
  embeddings (`decoder_pos_embed`), so the decoder keeps its training.
- The VAE encoder reads the action chunk. Its input projection
  (6 -> 7 columns) and its position encoding (102 -> 18 positions)
  start new.
- The action head (512 -> 7) starts new.
- Temporal ensembling is off. The executor joins the segments.
- Cube task: warm start from `act_all_ft50k`, 30k steps, batch 8.

### 3.2 Diffusion-BSP

- The U-Net works on a 1D sequence of `horizon` rows, with one
  channel for each action column. The first conv, its residual conv
  and the final conv change from 6 to 7 channels and start new. All
  other U-Net and ResNet weights are copied.
- Horizon 16 is divisible by 4, as the 3-level U-Net needs.
- `drop_n_last_frames = 0`: each target already holds the future, so
  every frame of an episode is a valid sample.
- LeRobot drops the first `n_obs_steps - 1` rows of the horizon at
  inference. A segment needs every row, so `segment.py` calls
  `diffusion.conditional_sample` directly. One live frame is repeated
  to fill the observation history (`n_obs_steps = 2`), as LeRobot
  does at the start of an episode.
- Parcel task:
  - The top camera is 848x480, and the wrist and side cameras are
    640x480. Diffusion stacks the cameras before the resize, so the
    shapes must be equal. `patch_diffusion()` and `_resize()` in
    `../parcel_task/train_mixed.py` resize each camera to 240x424
    first. Training and inference use the same patch.
  - AMP (mixed precision) is on, so 3 cameras fit in 6 GB.
  - Warm start from `diffusion_parcel_a1_80k`, 40k steps, batch 16.
- Problem: each plan starts from new random noise, so two plans
  0.3 s apart do not agree. The arm jumped 7-9 deg at each switch.
  Fix (cube task): `--ddim-steps 5 --max-step 6 --blend 0.2`. The
  direction reversals per second at 2x went from 8.3 to 2.5.

### 3.3 SmolVLA-BSP

- SmolVLA pads every action to `max_action_dim` (32) columns. The 7
  B-spline columns fit in this width, so no layer changes shape. All
  500 tensors are copied, and no layer starts new.
- `load_vlm_weights = False`: the VLM weights come from `--init`, not
  from the Hub base model.
- The processors come from `--init`: the task tokenizer and the camera
  rename (top -> camera1, wrist -> camera2, side -> camera3). Only the
  action normalizer and unnormalizer get the target stats.
- `empty_cameras = 0`, because all 3 pretrained camera slots have a
  real camera.
- The vision encoder stays frozen, and only the action expert trains
  (`train_expert_only`), as in the baseline.
- `noise=` lets the executor give a fixed start noise (as
  `--fixed-noise` does for Diffusion).
- Parcel task: warm start from `smolvla_parcel_a1_80k`, 20k steps,
  batch 12. The final loss is 0.131.

Known issue: `train_bsp.py` builds the model from the config, so 146M
VLM weights are float32 in place of bfloat16. Thus the checkpoint is
1198 MB, not 907 MB. Training ran at 3.2 s per step, compared with
1.2 s per step for `lerobot-train`. A possible fix: cast the VLM to
bfloat16 after the warm start. We did not test this fix.

## 4. Trained BSP models

| Model | Task | Init | Steps | Batch | Final loss | Time |
|---|---|---|---|---|---|---|
| `act_bsp_ws30k` | Cube | `act_all_ft50k` | 30k | 8 | 0.110 | ~5 h |
| `diffusion_bsp_ws40k` | Cube | `diffusion_all_100k` | 40k | 16 | 0.020 | ~4.6 h |
| `diffusion_bsp_parcel_40k` | Parcel | `diffusion_parcel_a1_80k` | 40k | 16 | 0.016 | ~5 h |
| `smolvla_bsp_parcel_20k` | Parcel | `smolvla_parcel_a1_80k` | 20k | 12 | 0.131 | ~18 h |

The loss values of different policies do not compare. Each policy
has its own loss.

## 5. Commands

```bash
# targets (once per dataset)
python outputs/portfolio/bsp/make_targets.py --dataset parcel-sort-a \
  --max-error 0.5

# training
bash outputs/portfolio/parcel_task/train_bsp_parcel.sh diffusion
bash outputs/portfolio/parcel_task/train_bsp_parcel.sh smolvla \
  outputs/train/smolvla_parcel_a1_80k/checkpoints/last/pretrained_model

# robot (cube task)
python outputs/portfolio/bsp/run_bsp.py --policy diffusion --spot SQ \
  --ddim-steps 5 --max-step 6 --blend 0.2 --speed 2 --episodes 5

# robot (parcel task, Diffusion-BSP)
bash outputs/portfolio/parcel_task/eval_parcel.sh diffusion_bsp \
  --level easy --episodes 2 --speed 1.5
```

`run_bsp_parcel.py` runs only Diffusion-BSP now. SmolVLA-BSP needs the
SmolVLA processors and the camera rename in this script.

## 6. Files

| File | Purpose |
|---|---|
| `bspline.py` | Fit, chunk, decode (from the official code) |
| `make_targets.py` | Targets for one dataset (`--scan` compares tolerances) |
| `train_bsp.py` | Training for ACT, Diffusion and SmolVLA |
| `segment.py` | One full segment from a policy |
| `run_bsp.py` | Robot executor, cube task (speed sweeps, debug logs) |
| `diagnose_bsp.py` | Table of the debug logs for each episode |
| `grid_executor.py` | Offline grid test of the executor settings |
| `../parcel_task/run_bsp_parcel.py` | Executor for the parcel task (3 cameras) |
