#!/usr/bin/env bash
# Control for the DAgger fine-tune: the same recipe as
# smolvla_parcel_b_dagger_trim_15k (lr 1e-4, warm-up 500, cosine over
# 15k steps), but on dataset B only (no corrections). Stops at 10k.
# Compare the start test at 5k and 10k with smolvla_b_dgt:005000/010000.
#   B only also loses the start motion -> the recipe is the cause
#   B only keeps it                    -> the corrections are the cause
# (The stats are the parcel-sort-b stats here, as in the old model.)
# Start:
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="control fine-tune" \
#     nohup bash parcel_vla/finetuning/queue_control_b.sh \
#     > outputs/train/queue_control_b.out 2>&1 &
set -uo pipefail

source "$(dirname "$0")/../env.sh"
T=$TRAIN
LOG=$T/queue_control_b.log
L=$LOW
S1=smolvla_parcel_b_control_10k
OUT=$T/$S1
INIT1=$T/smolvla_parcel_b_80k/checkpoints/last/pretrained_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

start_test() {
  # $1 = checkpoint step folder; runs once, after it is saved
  local ck=$OUT/checkpoints/$1/pretrained_model/model.safetensors
  while [ ! -e "$ck" ]; do sleep 60; done
  sleep 60
  note "start test $1"
  "$PY" -m parcel_vla.finetuning.start_test \
    "smolvla_b_ctl:$1" "smolvla_b_dgt:$1" >> "$LOG" 2>&1
  note "start test $1 done"
}

while pgrep -f "lerobot-train|train_mixed|train_bsp(\.py)? --policy" \
    >/dev/null; do
  sleep 60
done
note "queue start"
rm -rf "$OUT"
start_test 005000 &
"$LEROBOT_TRAIN" \
  --policy.path="$INIT1" \
  --dataset.repo_id=$HF_USER/parcel-sort-b \
  --dataset.root="$L/parcel-sort-b" \
  --rename_map="$RENAME" \
  --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=15000 \
  --batch_size=12 --num_workers=8 \
  --steps=10000 --save_freq=5000 --log_freq=200 \
  --wandb.enable=false \
  --job_name="$S1" --output_dir="$OUT" >> "$T/$S1.log" 2>&1
note "end $S1 (exit $?)"
wait
start_test 010000
note "queue end"
