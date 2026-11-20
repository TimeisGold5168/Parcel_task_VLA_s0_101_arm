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

Known issue (dataset A1 model): `train_bsp.py` built the model from the
config, so 146M VLM weights are float32 in place of bfloat16. Thus the
checkpoint is 1198 MB, not 907 MB. Training ran at 3.2 s per step,
compared with 1.2 s per step for `lerobot-train`. Fixed for dataset B:
`train_bsp.py` now casts the warm-start weights to the dtype of the
`--init` checkpoint.
At inference, `run_bsp_parcel.py` casts these weights to bfloat16:
the plan time goes from about 350 ms to 209 ms, and the offline error
does not change (2.93 deg).

Robot fixes for SmolVLA-BSP (parcel task, 1x):

| Settings | Result | Reversals/s | Acceleration p95 (deg/s^2) |
|---|---|---|---|
| float32 VLM, random noise, no blend (plan 325 ms) | 2 fails | 2.5-3.4 | 988-1410 |
| bfloat16 VLM, `--fixed-noise`, `--blend 0.2` (plan 223 ms) | 1 success (48.4 s) | 1.8 | 614 |
| SmolVLA 80k, chunks (reference) | Good | 1.1-1.2 | 441-450 |

With the old settings, the command was a sawtooth. The arm moved
inside a segment, then jumped back 7-10 deg at each switch. Fixed
noise cut the median jump between two plans from 6.2 to 4.0 deg
(offline).

## 4. Planning time, margin and the speed limit

### 4.1 The margin

The planner starts the next plan when the current segment is
`--replan-frac` (f) used. The new plan must arrive before the old
segment ends. If it arrives late, the arm stops at the end of the old
segment (a "hold"), then moves again. The time that is left on the
old segment when the new plan arrives is the margin:

```text
margin = (1 - f) * S / (30 * v) - L
```

- S = length of one segment (demo frames). For the parcel models,
  S is about 35-45 frames (1.2-1.5 s at 1x).
- f = `--replan-frac` (default 0.5).
- v = `--speed`.
- L = planning time (s).

The margin must stay above 0. Thus the speed limit is:

```text
v_max = (1 - f) * S / (30 * L)
```

Use the slow plans (p95 of L) and the short segments (small S). The
average values give a limit that is too high.

Example: SmolVLA-BSP, L = 0.223 s (robot), S = 40, f = 0.5:

| Speed | Margin | Result |
|---|---|---|
| 1x | 0.44 s | Safe |
| 1.5x | 0.22 s | Safe |
| 2x | 0.11 s | Tight: one slow plan or a short segment gives a hold |
| 2.5x | 0.04 s | Holds are likely |
| 3x | about 0 s | Holds |

With p95 L = 0.3 s and S = 33, v_max is about 1.8x. Diffusion-BSP
plans in about 95 ms, so its limit is about 2.3 times higher.

Other effects of L:

- The plan uses an observation that is L old. The executor starts the
  new segment at the point near "now" that is closest to the last
  command, so this delay does not cause a jump.
- A lower f gives more margin, but more segment switches.
- `--blend` must end before the old segment ends. The executor uses
  min(blend, time left), so a small margin also shortens the blend.

### 4.2 How to measure the limit on the robot

1. Run one episode at each speed with `--debug`:

   ```bash
   for S in 1 1.5 2; do
     bash outputs/portfolio/parcel_task/eval_parcel.sh smolvla_bsp \
       --level easy --episodes 1 --parcels 1 --parcel yellow_box \
       --where near_flip --time 60 --speed $S \
       --blend 0.2 --fixed-noise --debug
   done
   ```

2. Read the logs:

   ```bash
   python outputs/portfolio/bsp/diagnose_bsp.py outputs/portfolio/bsp/runs
   ```

| Column | Meaning | Target |
|---|---|---|
| `plan` | Mean and p95 planning time (ms) | p95 well below the time between switches |
| `margin` | Minimum time left on the old segment at a switch (s) | Above 0 |
| `hold` | % of ticks frozen at a segment end | About 0% |
| `seam` | p95 jump at a switch (deg) | Small (this is the jerk) |
| `lag` | Arm position compared with the command at plan time (deg) | Low |
| `rev/s` | Direction reversals per second | Near the chunk policy |

The limit of a model is the highest speed where `margin` stays above
0 and `hold` stays near 0%. A BSP claim needs the success rate and the
completion time at that speed, compared with the chunk policy at 1x.

### 4.3 Where the SmolVLA-BSP planning time goes

Profile on the laptop GPU (RTX PRO 500 Blackwell, 6 GB), 3 cameras,
one plan. Each part was timed with a CUDA sync, so the total is
higher than without the timer (about 210 ms offline, 223 ms on the
robot).

| Part | 10 steps | 5 steps | 3 steps |
|---|---|---|---|
| Images to the GPU + preprocessor | 2 ms | 2 ms | 2 ms |
| Vision encoder, 3 cameras | 73 ms | 73 ms | 73 ms |
| VLM prefix pass (16 layers) | 21 ms | 12 ms | 12 ms |
| Denoising steps (action expert) | 176 ms (10 x 17.6) | 50 ms (5 x 10) | 30 ms (3 x 10) |
| Total model | 287 ms | 141 ms | 122 ms |
| Error compared with the demo (episode 45) | 2.76 deg | 2.68 deg | 2.61 deg |

- The CPU part is about 2 ms. Thus CPU threads or SIMD do not help.
- The denoising steps are the largest part. Each step runs 16 tokens
  through 16 small layers. This is many small GPU kernels, so the
  time per kernel launch counts more than the compute.
- Fewer flow steps did not make the error worse on this episode
  (fixed noise). Use `--steps 5` on the robot to check the success
  rate and the smoothness.

The time for each step at 10 steps (17.6 ms) is higher than at 5
steps (10 ms). The cause is not known. A possible cause is a lower
GPU clock during the longer run.

### 4.4 How to make planning faster

| Method | Expected gain | Cost or risk |
|---|---|---|
| Fewer flow steps (`--steps 5`) | 287 -> 141 ms (measured) | Check success and smoothness on the robot |
| bfloat16 VLM (done in `run_bsp_parcel.py`) | 350 -> 209 ms (measured) | None found |
| CUDA graphs (`torch.compile(mode="reduce-overhead")`) on the denoising step | Large: launch overhead is the main cost of each step | The KV cache changes in place (`crop`); needs static shapes |
| Smaller camera images (512x512 per camera now) | Vision encoder time goes down | Needs retraining; less detail for the label |
| Fewer cameras | About 24 ms per camera | Needs retraining; less information |
| TensorRT or ONNX export | Large | Much work for a custom model |
| A separate process for the planner | Removes contention for the Python GIL with the control and camera threads (robot 223 ms, offline 209 ms) | More code (shared memory for images) |
| A faster GPU | Desktop GPUs are several times faster | Hardware cost |

More Python threads do not make one plan faster. The GPU runs one plan
at a time, and Python threads share one interpreter lock.

## 5. Trained BSP models

| Model | Task | Init | Steps | Batch | Final loss | Time |
|---|---|---|---|---|---|---|
| `act_bsp_ws30k` | Cube | `act_all_ft50k` | 30k | 8 | 0.110 | ~5 h |
| `diffusion_bsp_ws40k` | Cube | `diffusion_all_100k` | 40k | 16 | 0.020 | ~4.6 h |
| `diffusion_bsp_parcel_40k` | Parcel | `diffusion_parcel_a1_80k` | 40k | 16 | 0.016 | ~5 h |
| `smolvla_bsp_parcel_20k` | Parcel | `smolvla_parcel_a1_80k` | 20k | 12 | 0.131 | ~18 h |

The loss values of different policies do not compare. Each policy
has its own loss.

## 6. Commands

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

# robot (parcel task, SmolVLA-BSP)
bash outputs/portfolio/parcel_task/eval_parcel.sh smolvla_bsp \
  --level easy --episodes 2 --speed 1 --blend 0.2 --fixed-noise
```

`run_bsp_parcel.py --model smolvla_bsp` uses the processors saved in
the checkpoint (camera rename and tokenizer).

## 7. Files

| File | Purpose |
|---|---|
| `bspline.py` | Fit, chunk, decode (from the official code) |
| `make_targets.py` | Targets for one dataset (`--scan` compares tolerances) |
| `train_bsp.py` | Training for ACT, Diffusion and SmolVLA |
| `segment.py` | One full segment from a policy |
| `run_bsp.py` | Robot executor, cube task (speed sweeps, debug logs) |
| `diagnose_bsp.py` | Table of the debug logs for each episode |
| `grid_executor.py` | Offline grid test of the executor settings |
| `fast_denoise.py` | Compiled SmolVLA denoising step (static K/V cache) |
| `trt_plan.py` | TensorRT engines for the BSP planners |
| `cpp_control.py`, `ctl/` | C++ control process `bsp_ctl` and its Python side |
| `bench_plan.py`, `bench_control.py` | Plan and control benchmarks |
| `bench_summary.py`, `queue_bench.sh` | Benchmark queue and summary tables |
| `video_record.py`, `compile_showcase.py` | Showcase videos |
| `../parcel_task/run_bsp_parcel.py` | Executor for the parcel task (3 cameras) |
