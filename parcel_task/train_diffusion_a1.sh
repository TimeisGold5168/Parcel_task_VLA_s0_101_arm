#!/usr/bin/env bash
# Diffusion on the parcel dataset A1 (laptop GPU, 6 GB).
# Usage: bash outputs/portfolio/parcel_task/train_diffusion_a1.sh
#        RESUME=1 bash outputs/portfolio/parcel_task/train_diffusion_a1.sh
# Measured: 0.42 s/step at batch 16 with AMP -> 80k steps ~9.3 h.
# Checkpoints every 10k steps in outputs/train/$RUN/checkpoints.
set -euo pipefail

cd "$(dirname "$0")/../../.."
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
RUN=${RUN:-diffusion_parcel_a1_80k}
OUT=outputs/train/$RUN
ROOT=$HOME/.cache/huggingface/lerobot/Low101/parcel-sort-a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "${RESUME:-0}" = 1 ]; then
  exec "$PY" outputs/portfolio/parcel_task/train_mixed.py \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
fi

# Mixed camera sizes (top 848x480, wrist/side 640x480): train_mixed.py
# resizes every camera to resize_shape before Diffusion stacks them.
exec "$PY" outputs/portfolio/parcel_task/train_mixed.py \
  --policy.type=diffusion \
  --dataset.repo_id=Low101/parcel-sort-a \
  --dataset.root="$ROOT" \
  --policy.resize_shape="[240,424]" \
  --policy.crop_shape="[216,384]" \
  --policy.down_dims="[256,512,1024]" \
  --policy.use_amp=true \
  --policy.n_obs_steps=2 \
  --policy.horizon=64 \
  --policy.push_to_hub=false \
  --batch_size=16 \
  --num_workers=8 \
  --steps=80000 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --output_dir="$OUT"
