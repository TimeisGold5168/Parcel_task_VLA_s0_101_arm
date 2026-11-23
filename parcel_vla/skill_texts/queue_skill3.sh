#!/usr/bin/env bash
# Skill-text model for the language test. Offline only, no robot.
#   0. make_skill3.py [--variant distinct|distinct5] must have run
#      (dataset with the texts, outputs/train/skill3*_split.json)
#   1. train smolvla_parcel_b_sk3_10k (VARIANT=distinct:
#      smolvla_parcel_b_sk3d_10k, distinct5: smolvla_parcel_b_sk5d_10k)
#      from smolvla_parcel_b_80k, lr 1e-4, 10k steps, on the training
#      episodes only (held-out episodes out)
#   2. lang_test.py on 5k, 10k and the one-text control
#      (smolvla_b_all 10k) -> outputs/train/skill3_lang.md
#      (skill3d_lang.md)
# Times: training ~3.4 h, test ~10 min.
# Start (blocks sleep and the lid switch while it runs):
#   VARIANT=distinct systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="skill-text fine-tune" \
#     nohup bash parcel_vla/skill_texts/queue_skill3.sh \
#     > outputs/train/queue_skill3.out 2>&1 &
# Log: outputs/train/queue_skill3.log. A step that is done is skipped.
set -uo pipefail

source "$(dirname "$0")/../env.sh"
T=$TRAIN
L=$LOW
LOG=$T/queue_skill3.log
case "${VARIANT:-base}" in
  base) TAG=sk3 SPLIT=$T/skill3_split.json REPORT=$T/skill3_lang.md ;;
  distinct) TAG=sk3d SPLIT=$T/skill3d_split.json
            REPORT=$T/skill3d_lang.md ;;
  distinct5) TAG=sk5d SPLIT=$T/skill5d_split.json
             REPORT=$T/skill5d_lang.md ;;
  *) echo "VARIANT must be base, distinct or distinct5"; exit 1 ;;
esac
RUN=smolvla_parcel_b_${TAG}_10k
STEPS=10000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"
INIT=$T/smolvla_parcel_b_80k/checkpoints/last/pretrained_model

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }
done_ck() {
  [ -e "$T/$RUN/checkpoints/$1/training_state/training_step.json" ]
}

[ -e "$SPLIT" ] || fail "run make_skill3.py first ($SPLIT)"
read -r DS EPS < <("$PY" -c "import json
s = json.load(open('$SPLIT'))
print(s['dataset'], str(s['train_episodes']).replace(' ', ''))")

BUSY="lerobot-train|train_mixed|train_bsp(\.py)? --policy"
BUSY+="|rollout_mixed.py|run_bsp_parcel.py|record_preview.py"
while pgrep -f "$BUSY" >/dev/null; do
  sleep 60
done
note "queue start ($RUN, $DS)"

if ! done_ck "$(printf %06d $STEPS)"; then
  if done_ck 005000; then
    note "resume $RUN"
    CFG=$T/$RUN/checkpoints/last/pretrained_model/train_config.json
    "$LEROBOT_TRAIN" --resume=true --config_path="$CFG" \
      >> "$T/$RUN.log" 2>&1 || fail "training failed"
  else
    note "train $RUN"
    rm -rf "${T:?}/$RUN"
    "$LEROBOT_TRAIN" \
      --policy.path="$INIT" \
      --dataset.repo_id="$HF_USER/$DS" --dataset.root="$L/$DS" \
      --dataset.episodes="$EPS" \
      --rename_map="$RENAME" --policy.push_to_hub=false \
      --policy.optimizer_lr=1e-4 \
      --policy.scheduler_warmup_steps=500 \
      --policy.scheduler_decay_steps=$STEPS \
      --batch_size=12 --num_workers=4 \
      --steps=$STEPS --save_freq=5000 --log_freq=200 \
      --wandb.enable=false \
      --job_name="$RUN" --output_dir="$T/$RUN" >> "$T/$RUN.log" 2>&1 \
      || fail "training failed"
  fi
  note "end $RUN"
fi

if [ ! -e "$REPORT" ]; then
  note "language test ($REPORT)"
  "$PY" -m parcel_vla.skill_texts.lang_test "$RUN:005000" "$RUN:010000" \
    smolvla_parcel_b_all_20k:010000 --split "$SPLIT" --out "$REPORT" \
    > "${REPORT%.md}.log" 2>&1 || fail "language test failed"
fi
note "queue end"
