#!/usr/bin/env bash
# SmolVLA to 80k steps, then SmolVLA-BSP from it (laptop GPU).
#
#   nohup bash outputs/portfolio/parcel_task/queue_smolvla80k.sh \
#     > outputs/train/queue_smolvla80k.out 2>&1 &
#
# 1. smolvla_parcel_a1_80k: fine-tune from smolvla_parcel_a1_20k (step
#    20000) for 60k steps with a new LR schedule (warmup 1000, cosine
#    decay over 60k; the 20k run decays over 30k, so a plain resume to
#    80k would run 50k steps at the minimum LR). ~1.21 s/step -> ~20 h.
# 2. smolvla_bsp_parcel_20k from smolvla_parcel_a1_80k/last (~7 h).
# Robot policy runs need the GPU too: do not run them meanwhile.
# Resume step 1 after a stop: RESUME=1 RUN=smolvla_parcel_a1_80k
#   bash outputs/portfolio/parcel_task/train_smolvla_a1.sh
# Log: outputs/train/queue_smolvla80k.log
set -uo pipefail

cd "$(dirname "$0")/../../.."
TRAIN=/home/zhenting.lee/miniforge3/envs/lerobot/bin/lerobot-train
LOG=outputs/train/queue_smolvla80k.log
INIT=outputs/train/smolvla_parcel_a1_20k/checkpoints/020000/pretrained_model
RUN=smolvla_parcel_a1_80k
OUT=outputs/train/$RUN
ROOT=$HOME/.cache/huggingface/lerobot/Low101/parcel-sort-a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

while pgrep -f "lerobot-train|train_mixed.py|train_bsp.py --policy" \
    >/dev/null; do
  sleep 60
done

# Up to 3 attempts; an attempt after a crash resumes from the last
# checkpoint (at most save_freq steps are lost).
for TRY in 1 2 3; do
  [ -e "$OUT/checkpoints/060000" ] && break
  echo "$(date '+%F %T') start $RUN (try $TRY)" | tee -a "$LOG"
  if [ -e "$OUT/checkpoints/last" ]; then
    "$TRAIN" \
      --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true >> "$OUT.log" 2>&1
  else
    rm -rf "$OUT"
    "$TRAIN" \
      --policy.path="$INIT" \
      --dataset.repo_id=Low101/parcel-sort-a \
      --dataset.root="$ROOT" \
      --rename_map="$RENAME" \
      --policy.push_to_hub=false \
      --policy.scheduler_warmup_steps=1000 \
      --policy.scheduler_decay_steps=60000 \
      --batch_size=12 \
      --num_workers=8 \
      --steps=60000 \
      --save_freq=10000 \
      --log_freq=200 \
      --wandb.enable=false \
      --job_name="$RUN" \
      --output_dir="$OUT" >> "$OUT.log" 2>&1
  fi
  echo "$(date '+%F %T') end $RUN (exit $?)" | tee -a "$LOG"
  sleep 60
done

[ -e "$OUT/checkpoints/060000" ] || { echo "no 60k checkpoint"; exit 1; }
echo "$(date '+%F %T') start smolvla_bsp_parcel_20k" | tee -a "$LOG"
bash outputs/portfolio/parcel_task/train_bsp_parcel.sh smolvla \
  "$OUT/checkpoints/last/pretrained_model"
echo "$(date '+%F %T') end smolvla_bsp_parcel_20k" | tee -a "$LOG"
