#!/usr/bin/env bash
# Balanced all-skill fine-tune with an early check, then (if it passes
# the offline rules) its SmolVLA-BSP version. Offline only, no robot.
#   0. merge parcel-sort-b-all: each part one time (dataset B, far-edge,
#      box demos, stacked-parcel, black-box, on-table corrections)
#   1. run A: from smolvla_parcel_b_fb_15k/010000, lr 5e-5, 20k steps
#      (smolvla_parcel_b_all_20k). Stop at 10k, early_check.py on 5k
#      (log only) and 10k. If 10k passes, resume to 20k.
#   2. if 10k fails, run B: from smolvla_parcel_b_80k, lr 1e-4, 20k
#      steps (smolvla_parcel_b_allb_20k), checks logged, no stop
#   3. pick_checkpoint.py on the run -> outputs/train/skill_report_all.md
#   4. a checkpoint that passes rules 1-2 in all groups, with the lowest
#      mean new-skill fit; none -> stop (no BSP)
#   5. queue_bsp_skill.sh all from it (30k steps, report
#      outputs/train/skill_report_bsp_all.md)
# Times: 10k steps ~3.4 h, check ~15 min, report ~45 min, BSP ~11 h.
# Start (blocks sleep and the lid switch while it runs):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch \
#     --why="all-skill fine-tune" \
#     nohup bash outputs/portfolio/parcel_task/queue_all.sh \
#     > outputs/train/queue_all.out 2>&1 &
# Log: outputs/train/queue_all.log. A step that is done is skipped, so
# a second start continues.
set -uo pipefail

cd "$(dirname "$0")/../../.."
BIN=/home/zhenting.lee/miniforge3/envs/lerobot/bin
T=outputs/train
P=outputs/portfolio/parcel_task
L=$HOME/.cache/huggingface/lerobot/Low101
LOG=$T/queue_all.log
REPORT=$T/skill_report_all.md
DS=parcel-sort-b-all
PARTS=(parcel-sort-b rollout_dagger_far_edge_pull
       parcel-skill-boxstack-cut rollout_dagger_far_stack_cut
       rollout_dagger_black_box_cut rollout_dagger_on_table_cut)
STEPS=20000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=observation.images
RENAME="{\"$I.top\": \"$I.camera1\", \"$I.wrist\": \"$I.camera2\","
RENAME+=" \"$I.side\": \"$I.camera3\"}"

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { note "$*"; exit 1; }
done_ck() {
  [ -e "$T/$1/checkpoints/$2/training_state/training_step.json" ]
}

BUSY="lerobot-train|train_mixed.py|train_bsp.py --policy"
BUSY+="|rollout_mixed.py|run_bsp_parcel.py|record_preview.py"
while pgrep -f "$BUSY" >/dev/null; do
  sleep 60
done
note "queue start"

if [ ! -e "$L/$DS/meta/info.json" ]; then
  for p in "${PARTS[@]}"; do
    [ -e "$L/$p/meta/info.json" ] || fail "missing dataset $p"
  done
  note "merge ${PARTS[*]} -> $DS"
  IDS=$(printf "'Low101/%s', " "${PARTS[@]}")
  ROOTS=$(printf "'$L/%s', " "${PARTS[@]}")
  "$BIN/lerobot-edit-dataset" --new_repo_id "Low101/$DS" \
    --new_root "$L/$DS" --operation.type merge \
    --operation.repo_ids "[${IDS%, }]" \
    --operation.roots "[${ROOTS%, }]" >> "$LOG" 2>&1 \
    || fail "merge failed"
fi

# train RUN INIT LR [STOP]: to STEPS, or stop after checkpoint STOP
train() {
  local run=$1 init=$2 lr=$3 stop=${4:-} out=$T/$1 pid
  done_ck "$run" "$(printf %06d $STEPS)" && return 0
  [ -n "$stop" ] && done_ck "$run" "$stop" \
    && [ ! -e "$out/resume_ok" ] && return 0
  if done_ck "$run" 005000 || done_ck "$run" 010000; then
    note "resume $run"
    local cfg=$out/checkpoints/last/pretrained_model/train_config.json
    nice -n 15 "$BIN/lerobot-train" --resume=true --config_path="$cfg" \
      >> "$T/$run.log" 2>&1 &
  else
    note "train $run (init $init, lr $lr)"
    rm -rf "$out"
    nice -n 15 "$BIN/lerobot-train" \
      --policy.path="$init/pretrained_model" \
      --dataset.repo_id="Low101/$DS" --dataset.root="$L/$DS" \
      --rename_map="$RENAME" --policy.push_to_hub=false \
      --policy.optimizer_lr="$lr" \
      --policy.scheduler_warmup_steps=500 \
      --policy.scheduler_decay_steps=$STEPS \
      --batch_size=12 --num_workers=4 \
      --steps=$STEPS --save_freq=5000 --log_freq=200 \
      --wandb.enable=false \
      --job_name="$run" --output_dir="$out" >> "$T/$run.log" 2>&1 &
  fi
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -n "$stop" ] && [ ! -e "$out/resume_ok" ] \
        && done_ck "$run" "$stop" \
        && [ "$(readlink -f "$out/checkpoints/last")" \
             = "$(readlink -f "$out/checkpoints/$stop")" ]; then
      sleep 30
      note "stop $run at $stop for the early check"
      kill $pid; wait $pid 2>/dev/null
      pkill -f "job_name=$run"
      sleep 20
      return 0
    fi
    sleep 60
  done
  wait $pid
  note "end $run (exit $?)"
  done_ck "$run" "$(printf %06d $STEPS)"
}

check() {
  note "early check $1"
  "$BIN/python" $P/early_check.py "$1" >> "$LOG" 2>&1
  local r=$?
  note "early check $1: $([ $r = 0 ] && echo pass || echo fail)"
  return $r
}

RUN=smolvla_parcel_b_all_20k
SPEC=smolvla_b_all
if [ ! -e "$T/$RUN/resume_ok" ] && [ ! -e "$T/$RUN/switched" ]; then
  train $RUN $T/smolvla_parcel_b_fb_15k/checkpoints/010000 5e-5 010000 \
    || fail "run A failed before 10k"
  check $SPEC:005000
  if check $SPEC:010000; then
    touch "$T/$RUN/resume_ok"
  else
    touch "$T/$RUN/switched"
  fi
fi
if [ -e "$T/$RUN/resume_ok" ]; then
  train $RUN "" 5e-5 || fail "run A failed"
else
  note "run A failed the 10k check: switch to the base model"
  RUN=smolvla_parcel_b_allb_20k
  SPEC=smolvla_b_allb
  train $RUN $T/smolvla_parcel_b_80k/checkpoints/last 1e-4 \
    || fail "run B failed"
  check $SPEC:010000
  check $SPEC:020000
fi

if [ ! -e "$REPORT" ]; then
  note "offline report ($SPEC)"
  "$BIN/python" $P/pick_checkpoint.py --only "($SPEC)" \
    --out "$REPORT" > "$T/pick_all.log" 2>&1 || note "report failed"
fi

CK=$("$BIN/python" - "$REPORT" "$SPEC" <<'EOF'
import math, re, sys
rows = {}
try:
    text = open(sys.argv[1]).read()
except OSError:
    text = ""
pat = re.compile(r"\| `" + re.escape(sys.argv[2]) + r":(\d+)` \|(.*)\|")
for line in text.splitlines():
    m = pat.match(line)
    if not m:
        continue
    cells = [c.strip() for c in m.group(2).split("|")]
    fit, ok = float(cells[2]), cells[4] == "yes"
    r = rows.setdefault(m.group(1), {"ok": True, "fits": []})
    r["ok"] &= ok
    if not math.isnan(fit):
        r["fits"].append(fit)
good = {k: sum(v["fits"]) / len(v["fits"]) for k, v in rows.items()
        if v["ok"] and v["fits"]}
print(min(good, key=good.get) if good else "none")
EOF
)
if [ "$CK" = none ]; then
  note "no checkpoint passes rules 1-2 in all groups: no BSP"
  note "queue end"
  exit 0
fi
note "BSP init: $RUN/$CK"
INIT_RUN=$RUN INIT_CK=$CK bash $P/queue_bsp_skill.sh all
note "queue end"
