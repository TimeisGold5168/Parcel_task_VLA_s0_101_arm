#!/usr/bin/env bash
# Dataset B (Low101/parcel-sort-b, 99 episodes): 4 training jobs in a row
# on the laptop GPU. Measured on dataset B (Nov 5):
#   1. smolvla_parcel_b_80k        lerobot/smolvla_base, 80k, batch 12,
#                                  1.24 s/step                 ~27.5 h
#   2. smolvla_bsp_parcel_b_40k    from 1, 40k, batch 12, 1.27 s/step
#                                                               ~14 h
#   3. diffusion_parcel_b_80k      scratch, 80k, batch 16, AMP,
#                                  2.37 step/s                 ~9.4 h
#   4. diffusion_bsp_parcel_b_40k  from 3, 40k, batch 16        ~5 h
# Total about 56 h. Start (blocks sleep and the lid switch while it runs):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="dataset B training" \
#     nohup bash outputs/portfolio/parcel_task/queue_b.sh \
#     > outputs/train/queue_b.out 2>&1 &
# Progress: bash outputs/portfolio/parcel_task/queue_b.sh status
# A job that is done (final checkpoint exists) is skipped, so a second
# start continues the queue. Jobs 1 and 3 resume from their last
# checkpoint after a crash; jobs 2 and 4 start again from step 0.
# Robot policy runs need the GPU too: do not run them meanwhile.
set -uo pipefail

cd "$(dirname "$0")/../../.."
BIN=/home/zhenting.lee/miniforge3/envs/lerobot/bin
T=outputs/train
LOG=$T/queue_b.log
DS=parcel-sort-b
ROOT=$HOME/.cache/huggingface/lerobot/Low101/$DS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

S1=smolvla_parcel_b_80k
S2=smolvla_bsp_parcel_b_40k
S3=diffusion_parcel_b_80k
S4=diffusion_bsp_parcel_b_40k

if [ "${1:-}" = status ]; then
  tail -n 12 "$LOG" 2>/dev/null
  for r in $S1 $S3; do
    [ -e "$T/$r.log" ] && echo "$r: $(tr '\r' '\n' < "$T/$r.log" \
      | grep -o 'step:[0-9.K]* .*loss:[0-9.]*' | tail -1)"
  done
  for r in $S2 $S4; do
    [ -e "$T/$r.log" ] && echo "$r: $(grep '^step' "$T/$r.log" | tail -1)"
  done
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader
  exit 0
fi

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
smolvla_train() { "$BIN/lerobot-train" "$@"; }
mixed_train() {
  "$BIN/python" outputs/portfolio/parcel_task/train_mixed.py "$@"
}

while pgrep -f "lerobot-train|train_mixed.py|train_bsp.py --policy" \
    >/dev/null; do
  sleep 60
done

# lerobot-train style job: $1 run, $2 final step folder, $3 trainer
# function, rest = arguments for a fresh start. Up to 3 tries; a try
# after a crash resumes from the last checkpoint.
train_job() {
  local run=$1 final=$2 trainer=$3; shift 3
  local out=$T/$run
  local cfg=$out/checkpoints/last/pretrained_model/train_config.json
  for try in 1 2 3; do
    [ -e "$out/checkpoints/$final" ] && break
    note "start $run (try $try)"
    if [ -e "$out/checkpoints/last" ]; then
      "$trainer" --config_path="$cfg" --resume=true \
        >> "$T/$run.log" 2>&1
    else
      rm -rf "$out"
      "$trainer" "$@" --job_name="$run" --output_dir="$out" \
        >> "$T/$run.log" 2>&1
    fi
    note "end $run (exit $?)"
    sleep 60
  done
  [ -e "$out/checkpoints/$final" ]
}

# train_bsp.py job: $1 run, $2 final step folder, rest = arguments.
# Up to 2 tries, each from step 0 (train_bsp.py has no resume).
bsp_job() {
  local run=$1 final=$2; shift 2
  for try in 1 2; do
    [ -e "$T/$run/checkpoints/$final" ] && break
    note "start $run (try $try)"
    "$BIN/python" outputs/portfolio/bsp/train_bsp.py --dataset "$DS" \
      "$@" --out "$T/$run" > "$T/$run.log" 2>&1
    note "end $run (exit $?)"
    sleep 60
  done
  [ -e "$T/$run/checkpoints/$final" ]
}

note "queue start"

if train_job $S1 080000 smolvla_train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Low101/$DS --dataset.root="$ROOT" \
    --rename_map="$RENAME" \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=80000 \
    --batch_size=12 --num_workers=8 \
    --steps=80000 --save_freq=10000 --log_freq=200 \
    --wandb.enable=false; then
  bsp_job $S2 040000 --policy smolvla \
    --init "$T/$S1/checkpoints/last/pretrained_model" \
    --steps 40000 --batch 12 --workers 8 --save-freq 10000 \
    || note "$S2 failed"
else
  note "$S1 failed: skip $S2"
fi

if train_job $S3 080000 mixed_train \
    --policy.type=diffusion \
    --dataset.repo_id=Low101/$DS --dataset.root="$ROOT" \
    --policy.resize_shape="[240,424]" \
    --policy.crop_shape="[216,384]" \
    --policy.down_dims="[256,512,1024]" \
    --policy.use_amp=true \
    --policy.n_obs_steps=2 \
    --policy.horizon=64 \
    --policy.push_to_hub=false \
    --batch_size=16 --num_workers=8 \
    --steps=80000 --save_freq=10000 --log_freq=200 \
    --wandb.enable=false; then
  bsp_job $S4 040000 --policy diffusion \
    --init "$T/$S3/checkpoints/last/pretrained_model" \
    --steps 40000 --batch 16 --workers 8 --save-freq 10000 \
    || note "$S4 failed"
else
  note "$S3 failed: skip $S4"
fi

note "queue end"
