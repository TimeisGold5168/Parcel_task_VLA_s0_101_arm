#!/usr/bin/env bash
# DAgger round 1 for dataset B: merge the corrections into dataset B,
# then fine-tune SmolVLA B and SmolVLA-BSP B from their last checkpoints.
#   1. rollout_dagger_parcel_b without the intervention column
#   2. cut the idle home tail of each correction (trim_corrections.py)
#   3. merge with parcel-sort-b               -> parcel-sort-b-dagger-trim
#   4. B-spline targets for the merged dataset
#   5. smolvla_parcel_b_dagger_trim_15k      from smolvla_parcel_b_80k,
#                                            15k steps, batch 12  ~5.2 h
#   6. smolvla_bsp_parcel_b_dagger_trim_15k  from
#                                            smolvla_bsp_parcel_b_40k,
#                                            15k steps, batch 12  ~5.3 h
# 15k steps at batch 12 is about 1.3 passes over the merged dataset.
# The first runs (_dagger_15k, corrections not cut) did not leave the
# home pose: the correction tails made "at home: stay" the majority.
# Start (blocks sleep and the lid switch while it runs):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="DAgger fine-tune" \
#     nohup bash parcel_vla/finetuning/queue_dagger.sh [which] \
#     > outputs/train/queue_dagger.out 2>&1 &
#   which: smolvla (steps 1-5), bsp (1-4 and 6) or all (default)
# Progress: bash parcel_vla/finetuning/queue_dagger.sh status
# A step that is done is skipped, so a second start continues the queue.
# Robot policy runs need the GPU too: do not run them meanwhile.
set -uo pipefail

source "$(dirname "$0")/../env.sh"
T=$TRAIN
LOG=$T/queue_dagger.log
L=$LOW
COR=rollout_dagger_parcel_b
DS=parcel-sort-b-dagger-trim
WHICH=${1:-all}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

S1=smolvla_parcel_b_dagger_trim_15k
S2=smolvla_bsp_parcel_b_dagger_trim_15k
STEPS=15000
INIT1=$T/smolvla_parcel_b_80k/checkpoints/last/pretrained_model
INIT2=$T/smolvla_bsp_parcel_b_40k/checkpoints/last/pretrained_model

if [ "${1:-}" = status ]; then
  tail -n 12 "$LOG" 2>/dev/null
  [ -e "$T/$S1.log" ] && echo "$S1: $(tr '\r' '\n' < "$T/$S1.log" \
    | grep -o 'step:[0-9.K]* .*loss:[0-9.]*' | tail -1)"
  [ -e "$T/$S2.log" ] && echo "$S2: $(grep '^step' "$T/$S2.log" | tail -1)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader
  exit 0
fi

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }

while pgrep -f "lerobot-train|train_mixed|train_bsp(\.py)? --policy" \
    >/dev/null; do
  sleep 60
done

note "queue start"

if [ ! -e "$L/${COR}_noint/meta/info.json" ]; then
  note "remove the intervention column"
  lerobot-edit-dataset --repo_id "$HF_USER/$COR" --root "$L/$COR" \
    --new_repo_id "$HF_USER/${COR}_noint" --new_root "$L/${COR}_noint" \
    --operation.type remove_feature \
    --operation.feature_names "['intervention']" >> "$LOG" 2>&1 \
    || fail "remove_feature failed"
fi

if [ ! -e "$L/${COR}_trim/meta/info.json" ]; then
  note "cut the idle home tails -> ${COR}_trim"
  rm -rf "$L/${COR}_trim"
  "$PY" -m parcel_vla.finetuning.trim_corrections \
    --src "${COR}_noint" --dst "${COR}_trim" >> "$LOG" 2>&1 \
    || fail "trim failed"
fi

if [ ! -e "$L/$DS/meta/info.json" ]; then
  note "merge parcel-sort-b + ${COR}_trim -> $DS"
  lerobot-edit-dataset --new_repo_id "$HF_USER/$DS" \
    --new_root "$L/$DS" --operation.type merge \
    --operation.repo_ids "['$HF_USER/parcel-sort-b', '$HF_USER/${COR}_trim']" \
    --operation.roots "['$L/parcel-sort-b', '$L/${COR}_trim']" \
    >> "$LOG" 2>&1 || fail "merge failed"
fi

TG=$REPO_DIR/parcel_vla/bsp/targets_${DS}_e0.5.npz
if [ ! -e "$TG" ]; then
  note "B-spline targets for $DS"
  "$PY" -m parcel_vla.bsp.make_targets --dataset "$DS" \
    --max-error 0.5 >> "$LOG" 2>&1 || fail "make_targets failed"
fi

smolvla_job() {
  local out=$T/$S1
  local cfg=$out/checkpoints/last/pretrained_model/train_config.json
  for try in 1 2 3; do
    [ -e "$out/checkpoints/0$STEPS" ] && break
    note "start $S1 (try $try)"
    if [ -e "$out/checkpoints/last" ]; then
      "$LEROBOT_TRAIN" --config_path="$cfg" --resume=true \
        >> "$T/$S1.log" 2>&1
    else
      rm -rf "$out"
      "$LEROBOT_TRAIN" \
        --policy.path="$INIT1" \
        --dataset.repo_id="$HF_USER/$DS" --dataset.root="$L/$DS" \
        --rename_map="$RENAME" \
        --policy.push_to_hub=false \
        --policy.scheduler_warmup_steps=500 \
        --policy.scheduler_decay_steps=$STEPS \
        --batch_size=12 --num_workers=8 \
        --steps=$STEPS --save_freq=5000 --log_freq=200 \
        --wandb.enable=false \
        --job_name="$S1" --output_dir="$out" >> "$T/$S1.log" 2>&1
    fi
    note "end $S1 (exit $?)"
    sleep 60
  done
  [ -e "$out/checkpoints/0$STEPS" ] || note "$S1 failed"
}

bsp_job() {
  for try in 1 2; do
    [ -e "$T/$S2/checkpoints/0$STEPS" ] && break
    note "start $S2 (try $try)"
    "$PY" -m parcel_vla.bsp.train_bsp --policy smolvla \
      --dataset "$DS" \
      --init "$INIT2" \
      --steps $STEPS --batch 12 --workers 8 --save-freq 5000 \
      --out "$T/$S2" > "$T/$S2.log" 2>&1
    note "end $S2 (exit $?)"
    sleep 60
  done
  [ -e "$T/$S2/checkpoints/0$STEPS" ] || note "$S2 failed"
}

[ "$WHICH" != bsp ] && smolvla_job
[ "$WHICH" != smolvla ] && bsp_job

note "queue end"
