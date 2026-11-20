# lerobot-parcel-task

A personal project: a parcel sorting task with a LeRobot SO-101 arm,
trained by imitation learning. It compares ACT, Diffusion Policy and
SmolVLA, and the B-spline (BSP) version of each.

## Documents

| Document | Content |
|---|---|
| [`writeup.md`](writeup.md) | Full process and results: cube task, then parcel task |
| [`bsp/README.md`](bsp/README.md) | How we changed each policy to predict B-spline segments |
| [`test_protocol.md`](test_protocol.md) | Robot test protocol (cube task) |
| [`spot_test_plan.md`](spot_test_plan.md) | Spot test plan (cube task) |
| [`robot_tests_language_robustness.md`](robot_tests_language_robustness.md) | Language tests (cube task) |

## Project stages

1. Cube pick-and-place: ACT, Diffusion and SmolVLA on 3 datasets.
   Offline tests, robot spot tests, language and image tests.
2. B-spline execution (BSP): ACT-BSP and Diffusion-BSP. The arm runs
   faster and smoother.
3. Parcel task: ArUco labels, a 3-camera setup and a 100-episode
   dataset. Diffusion, Diffusion-BSP, SmolVLA and SmolVLA-BSP.
4. Next: a pick-and-place skill, then DAgger.

## Models (`models/`, Git LFS)

| Folder | Policy | Training |
|---|---|---|
| `diffusion_parcel_a1_80k` | Diffusion | 80k steps from scratch |
| `diffusion_bsp_parcel_40k` | Diffusion-BSP | 40k steps from `diffusion_parcel_a1_80k` |
| `smolvla_parcel_a1_80k` | SmolVLA | 80k steps from `lerobot/smolvla_base` |
| `smolvla_bsp_parcel_20k` | SmolVLA-BSP | 20k steps from `smolvla_parcel_a1_80k` |

All models use the dataset `Low101/parcel-sort-a` (private).
Each folder is a LeRobot `pretrained_model` folder. The BSP folders
also have `bsp.json`.

To get the weights:

1. Install Git LFS.
2. Run `git lfs install`.
3. Clone this repo, or run `git lfs pull` in a clone.

Without Git LFS, the `model.safetensors` files are small text pointers.

## Folders

| Folder | Content |
|---|---|
| `bsp/` | B-spline targets, training, executor and debug tools |
| `parcel_task/` | Recording, training, offline evaluation and robot runs for the parcel task |
| `parcel_labels/` | ArUco shipping labels and camera tests |
| `approach_demo/` | Camera layout and the approach demo |
| `image_perturbations/` | Image robustness test (cube task) |
| `spot_review/` | Spot test review (cube task) |
| `patches/` | Changes to LeRobot |
| `models/` | Model checkpoints (Git LFS) |

## Hardware

- SO-101 follower and leader arms.
- Cameras: RealSense (top), wrist camera, side camera.
- Training: laptop with an NVIDIA RTX PRO 500 Blackwell, 6 GB VRAM.
