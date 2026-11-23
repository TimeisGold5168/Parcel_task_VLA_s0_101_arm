#!/usr/bin/env bash
# Train Diffusion Policy from scratch on a 3-camera parcel dataset
# (laptop GPU, 6 GB).
#
#   bash parcel_vla/training/train_diffusion.sh
#   DATASET=<user>/<dataset> RUN=<run> \
#     bash parcel_vla/training/train_diffusion.sh
#   RESUME=1 RUN=<run> bash parcel_vla/training/train_diffusion.sh
#
# Default: the dataset B model, diffusion_parcel_b_80k (80k steps,
# batch 16, AMP). Measured: 2.37 step/s -> 80k steps ~9.4 h.
# Mixed camera sizes (top 848x480, wrist/side 640x480): train_mixed.py
# resizes every camera to resize_shape before Diffusion stacks them.
# Env: DATASET, RUN, STEPS, BATCH, SAVE_FREQ, DATA_ROOT (default: the
#   LeRobot cache of DATASET), TRAIN_DIR (outputs/train), PYTHON.
# Output: $TRAIN_DIR/$RUN/checkpoints/<step>/pretrained_model
set -euo pipefail

source "$(dirname "$0")/../env.sh"
DATASET=${DATASET:-$HF_USER/parcel-sort-b}
RUN=${RUN:-diffusion_parcel_b_80k}
OUT=$TRAIN/$RUN
ROOT=${DATA_ROOT:-$LEROBOT_HOME/$DATASET}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "${RESUME:-0}" = 1 ]; then
  exec "$PY" -m parcel_vla.training.train_mixed \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
fi

exec "$PY" -m parcel_vla.training.train_mixed \
  --policy.type=diffusion \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --policy.resize_shape="[240,424]" \
  --policy.crop_shape="[216,384]" \
  --policy.down_dims="[256,512,1024]" \
  --policy.use_amp=true \
  --policy.n_obs_steps=2 \
  --policy.horizon=64 \
  --policy.push_to_hub=false \
  --batch_size="${BATCH:-16}" \
  --num_workers=8 \
  --steps="${STEPS:-80000}" \
  --save_freq="${SAVE_FREQ:-10000}" \
  --log_freq=200 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --output_dir="$OUT"
