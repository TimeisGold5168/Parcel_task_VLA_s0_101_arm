#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on a 3-camera parcel dataset
# (laptop GPU, 6 GB).
#
#   bash parcel_vla/training/train_smolvla.sh
#   DATASET=<user>/<dataset> RUN=<run> bash parcel_vla/training/train_smolvla.sh
#   RESUME=1 RUN=<run> bash parcel_vla/training/train_smolvla.sh
#
# Default: the dataset B model, smolvla_parcel_b_80k (80k steps, batch
# 12, LR warmup 1000 steps, cosine decay over all steps).
# Measured: 1.24 s/step at batch 12 (4.7 GB) -> 80k steps ~27.5 h.
# Cameras map to the pretrained inputs: top/wrist/side -> camera1/2/3.
# Env: DATASET, RUN, STEPS, BATCH, SAVE_FREQ, DATA_ROOT (default: the
#   LeRobot cache of DATASET), TRAIN_DIR (outputs/train), LEROBOT_TRAIN.
# Output: $TRAIN_DIR/$RUN/checkpoints/<step>/pretrained_model
set -euo pipefail

source "$(dirname "$0")/../env.sh"
DATASET=${DATASET:-$HF_USER/parcel-sort-b}
RUN=${RUN:-smolvla_parcel_b_80k}
STEPS=${STEPS:-80000}
OUT=$TRAIN/$RUN
ROOT=${DATA_ROOT:-$LEROBOT_HOME/$DATASET}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

if [ "${RESUME:-0}" = 1 ]; then
  exec "$LEROBOT_TRAIN" \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
fi

exec "$LEROBOT_TRAIN" \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --rename_map="$RENAME" \
  --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --batch_size="${BATCH:-12}" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_freq="${SAVE_FREQ:-10000}" \
  --log_freq=200 \
  --wandb.enable=false \
  --job_name="$RUN" \
  --output_dir="$OUT"
