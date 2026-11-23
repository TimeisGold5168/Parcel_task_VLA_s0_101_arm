#!/usr/bin/env bash
# SmolVLA-BSP version of a new-skill SmolVLA B model (offline only).
#   bash parcel_vla/finetuning/queue_bsp_skill.sh [fb|fbs]
#   fb  (default): from smolvla_parcel_b_fb_15k/010000 on parcel-sort-b-fb
#   fbs:           from smolvla_parcel_b_fbs_10k/last on parcel-sort-b-fbs
#   all:           from $INIT_RUN/$INIT_CK (default
#                  smolvla_parcel_b_all_20k/last) on parcel-sort-b-all
#   Choose with the 3-parcel heap test of both models on the robot.
#   1. B-spline targets for the dataset (make_targets.py)
#   2. smolvla_bsp_parcel_b_<skill>_30k, 30k steps, batch 12 (~10.3 h at
#      0.81 step/s). Same recipe as smolvla_bsp_parcel_b_40k (from the
#      SmolVLA B model): warm-up 500, cosine decay over the full run.
#      --decay is needed: the init configs decay over 10-15k steps and
#      would leave the rest of the run at the low LR.
#   3. offline report: pick_checkpoint.py (start test + grasp fit on the
#      new-skill demos and dataset B), reference smolvla_bsp_b
#      -> outputs/train/skill_report_bsp_<skill>.md
# Start (blocks sleep and the lid switch while it runs):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="BSP skill" \
#     nohup bash parcel_vla/finetuning/queue_bsp_skill.sh fb \
#     > outputs/train/queue_bsp_fb.out 2>&1 &
# A step that is done is skipped, so a second start continues the queue.
# Checkpoints every 5k steps. Stop early (keeps them in place):
#   pkill -f queue_all.sh; pkill -f queue_bsp_skill; pkill -f train_bsp.py
# A later start moves a stopped run to <run>_stopped_<date> first,
# because train_bsp.py has no resume.
set -uo pipefail

source "$(dirname "$0")/../env.sh"
T=$TRAIN
SKILL=${1:-fb}
case "$SKILL" in
  fb)  INIT=$T/smolvla_parcel_b_fb_15k/checkpoints/010000 ;;
  fbs) INIT=$T/smolvla_parcel_b_fbs_10k/checkpoints/last ;;
  all) INIT=$T/${INIT_RUN:-smolvla_parcel_b_all_20k}
       INIT=$INIT/checkpoints/${INIT_CK:-last} ;;
  *) echo "usage: $0 [fb|fbs|all]"; exit 1 ;;
esac
INIT=$INIT/pretrained_model
LOG=$T/queue_bsp_$SKILL.log
DS=parcel-sort-b-$SKILL
S=smolvla_bsp_parcel_b_${SKILL}_30k
STEPS=30000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }

while pgrep -f "lerobot-train|train_mixed|train_bsp(\.py)? --policy" \
    >/dev/null; do
  sleep 60
done

note "queue start ($SKILL)"

TG=$REPO_DIR/parcel_vla/bsp/targets_${DS}_e0.5.npz
if [ ! -e "$TG" ]; then
  note "B-spline targets for $DS"
  "$PY" -m parcel_vla.bsp.make_targets --dataset "$DS" \
    --max-error 0.5 >> "$LOG" 2>&1 || fail "make_targets failed"
fi

for try in 1 2; do
  [ -e "$T/$S/checkpoints/0$STEPS" ] && break
  if compgen -G "$T/$S/checkpoints/0[0-9]*" >/dev/null; then
    # train_bsp.py has no resume: keep the checkpoints of a stopped run
    OLD=$T/${S}_stopped_$(date +%m%d_%H%M)
    mv "$T/$S" "$OLD"
    mv "$T/$S.log" "$OLD.log" 2>/dev/null
    note "kept the stopped run in $OLD"
    [ "$try" = 2 ] && fail "$S stopped after a checkpoint, no retry"
  fi
  note "start $S (try $try)"
  "$PY" -m parcel_vla.bsp.train_bsp --policy smolvla \
    --dataset "$DS" --init "$INIT" \
    --steps $STEPS --warmup 500 --decay $STEPS \
    --batch 12 --workers 8 --save-freq 5000 \
    --out "$T/$S" > "$T/$S.log" 2>&1
  note "end $S (exit $?)"
  sleep 60
done
[ -e "$T/$S/checkpoints/0$STEPS" ] || fail "$S failed"

note "offline report"
"$PY" -m parcel_vla.finetuning.pick_checkpoint \
  --only "smolvla_bsp_b_$SKILL)" --out "$T/skill_report_bsp_$SKILL.md" \
  > "$T/pick_bsp_$SKILL.log" 2>&1 || note "report failed"

note "queue end"
