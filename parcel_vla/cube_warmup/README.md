# cube_warmup (before phase 01)

This folder holds the first task: "pick green cube and place into the
bin", with ACT, Diffusion Policy and SmolVLA. The parcel task started
from its lessons. The test protocols are in [`docs/`](../../docs).

| File | Purpose |
|---|---|
| `run_robot_test.sh` | Robot test of one cube model |
| `run_spot.sh` | One test spot for every model, then auto-score |
| `run_configs.sh` | Inference-config comparison on one spot |
| `review_rollouts.py` | Auto-score spot-test rollouts and build review sheets |
| `vlm_frames.py` | Frame grids for a vision-model judge |
| `train_overnight.sh` | ACT-BSP, then Diffusion-BSP (warm start) |
| `colab_act_finetune.ipynb`, `upload_for_colab.sh` | ACT fine-tune on Google Colab, and the upload it needs |

The cube scripts use a front camera and a wrist camera
(`FRONT_DEV`, `WRIST_DEV`). The review sheets and the score tables go
to `runs/`. `run_configs.sh` reads its config list from
`runs/results_configs.csv`. That table is not in this repository.

## Lessons for the parcel task

1. ACT ran its 100-step chunk blind and jumped up to 69.9°. Temporal
   ensembling cut the max jump to 3.1°.
2. A start pose outside the demos and a layout that changed in every
   demo made the policy ignore the cameras. A fixed home pose, tape
   marks and new demos fixed it.
3. A B-spline policy runs the same curve at 1×, 2× or 3×: back home in
   4.5 s at 3×, against 11.9 s for the normal model.
4. A blue cube on a trained spot pulled SmolVLA (0/6), not ACT or
   Diffusion (6/7). The results are in
   [`results/cube_warmup/`](../../results/cube_warmup).
