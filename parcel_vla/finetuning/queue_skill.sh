#!/usr/bin/env bash
# New-skill fine-tunes of SmolVLA B, one model per skill, then both:
#   fe  = parcel-sort-b + rollout_dagger_far_edge_pull  -> parcel-sort-b-fe
#   box = parcel-sort-b + parcel-skill-boxstack-cut -> parcel-sort-b-box
#   fb  = parcel-sort-b + far edge + box demos 3x   -> parcel-sort-b-fb
#   fbs = fb + stacked-parcel DAgger picks          -> parcel-sort-b-fbs
#     (INIT=outputs/train/smolvla_parcel_b_fb_15k INIT_CK=010000
#      STEPS=10000 RUN_NAME=smolvla_parcel_b_fbs_10k)
#   all = parcel-sort-b + far edge + box demos 1x + stacked-parcel,
#         black-box and on-table corrections, nothing repeated
#                                                    -> parcel-sort-b-all
#     (INIT=outputs/train/smolvla_parcel_b_fb_15k INIT_CK=010000
#      STEPS=20000 LR=5e-5)
# LR (optional) sets --policy.optimizer_lr (default: the init config).
# Recipe of the control run (smolvla_parcel_b_control_10k), which kept
# the start motion: from smolvla_parcel_b_80k, lr 1e-4, warm-up 500,
# batch 12, 5k steps (about 1.7 h). Only the data changes.
# STEPS=15000 for fb: with the box demos 3x, each box frame is seen
# about 4 times, not 0.7 (the box model at 5k took the bottom box),
# about 5.2 h, saves every 5k.
# INIT=<run folder> starts from that run (default smolvla_parcel_b_80k);
# RUN_NAME and SPEC (offline_eval_b.SPECS name) then name the new run.
# After training: start_test.py on the new model and the old model.
# Start (blocks sleep and the lid switch while it runs):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="skill fine-tune" \
#     nohup bash parcel_vla/finetuning/queue_skill.sh fe \
#     > outputs/train/queue_skill_fe.out 2>&1 &
# Progress: bash parcel_vla/finetuning/queue_skill.sh status
# NICE=1 (default) runs at low CPU priority with 4 data workers, so a
# recording at the same time keeps its frame rate.
set -uo pipefail

source "$(dirname "$0")/../env.sh"
T=$TRAIN
LOG=$T/queue_skill.log
L=$LOW
SKILL=${1:?usage: queue_skill.sh fe|box|fb|fbs|status}
STEPS=${STEPS:-5000}
SAVE=2500
[ "$STEPS" -ge 15000 ] && SAVE=5000
LAST=$(printf %06d "$STEPS")
INIT1=${INIT:-$T/smolvla_parcel_b_80k}/checkpoints/${INIT_CK:-last}
INIT1=$INIT1/pretrained_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

if [ "$SKILL" = status ]; then
  tail -n 12 "$LOG" 2>/dev/null
  for f in "$T"/smolvla_parcel_b_{fe,box,fb,fbs,all}_*k.log; do
    [ -e "$f" ] && echo "$(basename "$f" .log): $(tr '\r' '\n' < "$f" \
      | grep -o 'step:[0-9.K]* .*loss:[0-9.]*' | tail -1)"
  done
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader
  exit 0
fi

case "$SKILL" in
  fe)  PARTS=(parcel-sort-b rollout_dagger_far_edge_pull) ;;
  box) PARTS=(parcel-sort-b parcel-skill-boxstack-cut) ;;
  fb)  PARTS=(parcel-sort-b rollout_dagger_far_edge_pull
              parcel-skill-boxstack-cut parcel-skill-boxstack-cut
              parcel-skill-boxstack-cut) ;;
  fbs) PARTS=(parcel-sort-b rollout_dagger_far_edge_pull
              parcel-skill-boxstack-cut parcel-skill-boxstack-cut
              parcel-skill-boxstack-cut rollout_dagger_far_stack_pull) ;;
  all) PARTS=(parcel-sort-b rollout_dagger_far_edge_pull
              parcel-skill-boxstack-cut rollout_dagger_far_stack_cut
              rollout_dagger_black_box_cut
              rollout_dagger_on_table_cut) ;;
  *) echo "unknown skill $SKILL"; exit 1 ;;
esac
DS=parcel-sort-b-$SKILL
RUN=${RUN_NAME:-smolvla_parcel_b_${SKILL}_$((STEPS / 1000))k}
SPEC=${SPEC:-smolvla_b_$SKILL}
OUT=$T/$RUN

note() { echo "$(date '+%F %T') [$SKILL] $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }

for p in "${PARTS[@]}"; do
  [ -e "$L/$p/meta/info.json" ] || fail "missing dataset $p"
done

BUSY="lerobot-train|train_mixed|train_bsp(\.py)? --policy"
BUSY+="|rollout_mixed.py|run_bsp_parcel.py|record_preview.py"
while pgrep -f "$BUSY" >/dev/null; do
  sleep 60
done
note "queue start"

if [ ! -e "$L/$DS/meta/info.json" ]; then
  note "merge ${PARTS[*]} -> $DS"
  rm -rf "$L/$DS"
  IDS=$(printf "'$HF_USER/%s', " "${PARTS[@]}")
  ROOTS=$(printf "'$L/%s', " "${PARTS[@]}")
  lerobot-edit-dataset --new_repo_id "$HF_USER/$DS" \
    --new_root "$L/$DS" --operation.type merge \
    --operation.repo_ids "[${IDS%, }]" \
    --operation.roots "[${ROOTS%, }]" \
    >> "$LOG" 2>&1 || fail "merge failed"
fi

PRE=()
WORKERS=8
LR_ARG=()
[ -n "${LR:-}" ] && LR_ARG=(--policy.optimizer_lr="$LR")
if [ "${NICE:-1}" = 1 ]; then
  PRE=(nice -n 15)
  WORKERS=4
fi

if [ ! -e "$OUT/checkpoints/$LAST/pretrained_model/model.safetensors" ]
then
  note "train $RUN on $DS"
  rm -rf "$OUT"
  "${PRE[@]}" "$LEROBOT_TRAIN" \
    --policy.path="$INIT1" \
    --dataset.repo_id="$HF_USER/$DS" \
    --dataset.root="$L/$DS" \
    --rename_map="$RENAME" \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=$STEPS "${LR_ARG[@]}" \
    --batch_size=12 --num_workers=$WORKERS \
    --steps=$STEPS --save_freq=$SAVE --log_freq=200 \
    --wandb.enable=false \
    --job_name="$RUN" --output_dir="$OUT" >> "$T/$RUN.log" 2>&1 \
    || fail "train failed"
fi

note "start test"
CKS=()
for ck in "$OUT"/checkpoints/[0-9]*; do
  CKS+=("$SPEC:$(basename "$ck")")
done
"$PY" -m parcel_vla.finetuning.start_test \
  "${CKS[@]}" smolvla_b >> "$LOG" 2>&1
note "queue end"
