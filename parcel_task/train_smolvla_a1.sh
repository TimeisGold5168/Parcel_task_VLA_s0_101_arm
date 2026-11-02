#!/usr/bin/env bash
# SmolVLA on the parcel dataset A1 (laptop GPU, 6 GB).
# Usage: bash outputs/portfolio/parcel_task/train_smolvla_a1.sh
#        RESUME=1 bash outputs/portfolio/parcel_task/train_smolvla_a1.sh
# Measured: 1.27 s/step at batch 12 (4.7 GB) -> 20k steps ~7.1 h.
# Cameras map to the pretrained inputs: top/wrist/side -> camera1/2/3.
set -euo pipefail

cd "$(dirname "$0")/../../.."
TRAIN=/home/zhenting.lee/miniforge3/envs/lerobot/bin/lerobot-train
RUN=${RUN:-smolvla_parcel_a1_20k}
OUT=outputs/train/$RUN
ROOT=$HOME/.cache/huggingface/lerobot/Low101/parcel-sort-a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

if [ "${RESUME:-0}" = 1 ]; then
  exec "$TRAIN" \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
fi

exec "$TRAIN" \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=Low101/parcel-sort-a \
  --dataset.root="$ROOT" \
  --rename_map="$RENAME" \
  --policy.push_to_hub=false \
  --batch_size=12 \
  --num_workers=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --output_dir="$OUT"
