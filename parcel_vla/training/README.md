# training (phase 03)

This folder holds the training scripts of the two policies. The
defaults reproduce the dataset B models on a 6 GB laptop GPU.

| File | Purpose |
|---|---|
| `train_smolvla.sh` | Fine-tune `lerobot/smolvla_base` (action expert only) |
| `train_diffusion.sh` | Train Diffusion Policy from scratch (through `train_mixed.py`) |
| `train_mixed.py` | `lerobot-train` for Diffusion with cameras of different sizes |
| `colab_smolvla_parcel.ipynb` | SmolVLA training on Google Colab (dataset A1) |

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `$HF_USER/parcel-sort-b` | Hub id of the dataset |
| `RUN` | `smolvla_parcel_b_80k`, `diffusion_parcel_b_80k` | Run name |
| `STEPS` | 80000 | Training steps |
| `BATCH` | 12 (SmolVLA), 16 (Diffusion) | Batch size |
| `SAVE_FREQ` | 10000 | Steps between checkpoints |
| `TRAIN_DIR` | `outputs/train` | Folder of the runs |
| `RESUME` | 0 | 1 continues the run from its last checkpoint |

```bash
DATASET=$HF_USER/parcel-sort-b RUN=smolvla_parcel_b_80k \
  bash parcel_vla/training/train_smolvla.sh
DATASET=$HF_USER/parcel-sort-b RUN=diffusion_parcel_b_80k \
  bash parcel_vla/training/train_diffusion.sh
```

Measured on the 6 GB laptop GPU:

| Run | Steps | Batch | Speed | Time | GPU memory |
|---|---|---|---|---|---|
| SmolVLA | 80k | 12 | 1.24 s/step | 27.4 h | 4.7 GB |
| Diffusion (AMP) | 80k | 16 | 2.37 steps/s | 9.4 h | fits 6 GB |

## Notes

- SmolVLA has three pretrained camera inputs, `camera1` to `camera3`.
  The scripts rename `top`, `wrist` and `side` to them.
- Diffusion stacks the camera images before it resizes them. The top
  camera (848×480) and the other cameras (640×480) have different
  sizes, so `train_mixed.py` resizes each camera to 240×424 first. The
  robot runs use the same patch (`patch_diffusion()`).
- The SmolVLA LR schedule must decay over the full run. The script
  sets `--policy.scheduler_decay_steps` to `STEPS`. With the default
  decay of 30k steps, an 80k run spends 50k steps at the minimum LR.
- To train with the LeRobot command line only, see the main
  [README](../../README.md#b-with-the-lerobot-command-line-only).
