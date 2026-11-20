#!/usr/bin/env bash
# DAgger round 1, gentle fine-tune of SmolVLA B. The first fine-tunes
# (queue_dagger.sh: lr 1e-4, merged-data stats) lost the start motion
# from the home pose (wrist roll turn, gripper open) by 5k-10k steps.
#   1. merge parcel-sort-b twice + rollout_dagger_parcel_b_trim
#      -> parcel-sort-b2x-dagger-trim (B episodes sampled 2x)
#   2. stats.json of that dataset = stats.json of parcel-sort-b, so
#      the model keeps its old normalization
#   3. smolvla_parcel_b_dagger_gentle_6k from smolvla_parcel_b_80k:
#      lr 1e-5 (warm-up 200, cosine to 1e-6), 6k steps, batch 12 ~2.1 h
#   4. offline start test (/tmp/h5/start_test.py style) on the result
# Start:
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="DAgger gentle fine-tune" \
#     nohup bash outputs/portfolio/parcel_task/queue_dagger_gentle.sh \
#     > outputs/train/queue_dagger_gentle.out 2>&1 &
# Robot policy runs need the GPU too: do not run them meanwhile.
set -uo pipefail

cd "$(dirname "$0")/../../.."
BIN=/home/zhenting.lee/miniforge3/envs/lerobot/bin
T=outputs/train
LOG=$T/queue_dagger_gentle.log
L=$HOME/.cache/huggingface/lerobot/Low101
COR=rollout_dagger_parcel_b_trim
DS=parcel-sort-b2x-dagger-trim
S1=smolvla_parcel_b_dagger_gentle_6k
STEPS=6000
INIT1=$T/smolvla_parcel_b_80k/checkpoints/last/pretrained_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }

while pgrep -f "lerobot-train|train_mixed.py|train_bsp.py --policy" \
    >/dev/null; do
  sleep 60
done
note "queue start"

if [ ! -e "$L/$DS/meta/info.json" ]; then
  note "merge parcel-sort-b x2 + $COR -> $DS"
  rm -rf "$L/$DS"
  "$BIN/lerobot-edit-dataset" --new_repo_id "Low101/$DS" \
    --new_root "$L/$DS" --operation.type merge \
    --operation.repo_ids \
    "['Low101/parcel-sort-b', 'Low101/parcel-sort-b', 'Low101/$COR']" \
    --operation.roots \
    "['$L/parcel-sort-b', '$L/parcel-sort-b', '$L/$COR']" \
    >> "$LOG" 2>&1 || fail "merge failed"
  cp "$L/$DS/meta/stats.json" "$L/$DS/meta/stats_merged.json"
  cp "$L/parcel-sort-b/meta/stats.json" "$L/$DS/meta/stats.json"
  note "stats.json = parcel-sort-b (merged stats in stats_merged.json)"
fi

out=$T/$S1
if [ ! -e "$out/checkpoints/00$STEPS" ]; then
  note "start $S1"
  rm -rf "$out"
  "$BIN/lerobot-train" \
    --policy.path="$INIT1" \
    --dataset.repo_id="Low101/$DS" --dataset.root="$L/$DS" \
    --rename_map="$RENAME" \
    --policy.push_to_hub=false \
    --policy.optimizer_lr=1e-5 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps=$STEPS \
    --policy.scheduler_decay_lr=1e-6 \
    --batch_size=12 --num_workers=8 \
    --steps=$STEPS --save_freq=2000 --log_freq=200 \
    --wandb.enable=false \
    --job_name="$S1" --output_dir="$out" >> "$T/$S1.log" 2>&1
  note "end $S1 (exit $?)"
fi
[ -e "$out/checkpoints/00$STEPS" ] || fail "$S1 failed"

note "offline start test"
"$BIN/python" outputs/portfolio/parcel_task/start_test.py \
  smolvla_b smolvla_b_dgt smolvla_b_dgg:002000 smolvla_b_dgg:004000 \
  smolvla_b_dgg >> "$LOG" 2>&1 || note "start test failed"
note "queue end"
