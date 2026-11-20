#!/usr/bin/env bash
# DAgger (human in the loop) corrections for a dataset B policy.
#
#   bash outputs/portfolio/parcel_task/dagger_parcel.sh <model> <n> [tag]
#
#   model  smolvla_b (RTC flags below) | diffusion_b |
#          smolvla_b_fb (box + far edge, CKPT_STEP default 010000)
#   REPO=Low101/... records to another dataset
#   n      correction episodes to record in this session
#   tag    failure type, log only: far_edge, on_table, black_box,
#          box_stack (one box on another box: take the top box first)
#
# The policy drives the follower. Only the corrections are recorded:
# each correction (Tab to Tab) is one episode of
#   ~/.cache/huggingface/lerobot/Low101/rollout_dagger_parcel_b
# with intervention=True. Later sessions add episodes (resume).
# Keys (global, X11; the terminal or any window can have the focus):
#   Space = pause the policy. The leader arm moves to the follower pose.
#   Tab   = start the correction: the leader is free, the follower
#           copies it, frames are recorded. Tab again = stop and save.
#   Space = give control back to the policy (from pause).
#   Esc   = end the session.
# Correction rule: start from the bad state, finish the parcel (grasp,
# flip if the label is up, release label down, exit push), go back to
# the ready pose, then Tab. The demos end at the ready pose, and in
# the pause the follower does not copy the leader.
# Reset: there is no timed reset. After the save, the robot holds its
# pose and the policy is stopped; place the next parcel, then Space.
# To reset after a policy success, Space (pause) first.
# SESSION_TIME (s, default 3600, includes the resets) limits the session.
set -euo pipefail

MODEL=${1:?model: smolvla_b or diffusion_b}
N=${2:?number of corrections}
TAG=${3:-}
TASK="sort the parcel so the label faces down and push it off the table"
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$HERE/../../train
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python

EXTRA=()
case "$MODEL" in
  smolvla_b|smolvla_b_fb)
    POLICY=$TRAIN/smolvla_parcel_b_80k/checkpoints/last/pretrained_model
    [ "$MODEL" = smolvla_b_fb ] && POLICY=$TRAIN/smolvla_parcel_b_fb_15k/\
checkpoints/${CKPT_STEP:-010000}/pretrained_model
    MAP='{"observation.images.top": "observation.images.camera1",'
    MAP+=' "observation.images.wrist": "observation.images.camera2",'
    MAP+=' "observation.images.side": "observation.images.camera3"}'
    EXTRA=(--rename_map="$MAP"
           --inference.type=rtc --inference.rtc.execution_horizon=10
           --inference.queue_threshold=30)
    export SMOOTH=${SMOOTH:-7} FREEZE=${FREEZE:-1}
    export RTC_EXACT_DELAY=${RTC_EXACT_DELAY:-1} COMPILE=${COMPILE:-default}
    ;;
  diffusion_b)
    POLICY=$TRAIN/diffusion_parcel_b_80k/checkpoints/last/pretrained_model
    EXTRA=(--policy.noise_scheduler_type=DDIM
           --policy.num_inference_steps="${STEPS:-10}")
    ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
POLICY=${POLICY_PATH:-$POLICY}
export PREVIEW_LABEL="DAgger $MODEL ${TAG}"

V=/dev/v4l/by-id
RS=""
for d in /sys/class/video4linux/video*; do
  grep -q RealSense "$d/name" || continue
  if v4l2-ctl -d "/dev/${d##*/}" --list-formats 2>/dev/null \
      | grep -q YUYV; then
    RS=/dev/${d##*/}
    break
  fi
done
WRIST=$V/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607053-video-index0
SIDE=$V/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2607071-video-index0
S=/dev/serial/by-id
FOLLOWER=$S/usb-1a86_USB_Single_Serial_5B61034620-if00
LEADER=$S/usb-1a86_USB_Single_Serial_5B79016363-if00
for d in "$RS" "$WRIST" "$SIDE" "$FOLLOWER" "$LEADER"; do
  [ -e "$d" ] || { echo "missing device: $d"; exit 1; }
done

CAMERAS="{ \
top: {type: opencv, index_or_path: $RS, width: 848, \
height: 480, fps: 30, fourcc: YUYV}, \
wrist: {type: opencv, index_or_path: $WRIST, width: 640, \
height: 480, fps: 30, fourcc: MJPG}, \
side: {type: opencv, index_or_path: $SIDE, width: 640, \
height: 480, fps: 30, fourcc: MJPG}}"

REPO=${REPO:-Low101/rollout_dagger_parcel_b}
ROOT=$HOME/.cache/huggingface/lerobot/$REPO
RESUME=false
[ -d "$ROOT/meta/episodes" ] && RESUME=true
BEFORE=0
[ -f "$ROOT/meta/info.json" ] && BEFORE=$("$PY" -c "import json, sys; \
print(json.load(open(sys.argv[1]))['total_episodes'])" "$ROOT/meta/info.json")

echo "DAgger $MODEL: $N corrections, tag=${TAG:-none}, resume=$RESUME"
echo "dataset: $ROOT ($BEFORE episodes now)"
echo "Space = pause, Tab = correct (Tab again = save), Esc = end"
read -rp "Robot at home, heap built? Press Enter to start. "

STATUS=0
"$PY" "$HERE/rollout_mixed.py" \
  --strategy.type=dagger \
  --strategy.num_episodes="$N" \
  --strategy.record_autonomous=false \
  --policy.path="$POLICY" \
  "${EXTRA[@]}" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER" \
  --robot.id=my_follower_arm \
  --robot.num_read_retries=10 \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER" \
  --teleop.id=my_leader_arm \
  --teleop.num_read_retries=10 \
  --task="$TASK" \
  --duration="${SESSION_TIME:-3600}" \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$N" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_queue_maxsize=150 \
  --display_data=false \
  --resume="$RESUME" || STATUS=$?

AFTER=$BEFORE
[ -f "$ROOT/meta/info.json" ] && AFTER=$("$PY" -c "import json, sys; \
print(json.load(open(sys.argv[1]))['total_episodes'])" "$ROOT/meta/info.json")
LOG=$ROOT/dagger_log.csv
if [ "$AFTER" -gt "$BEFORE" ]; then
  [ -f "$LOG" ] || echo "time,model,checkpoint,episodes,tag" > "$LOG"
  echo "$(date +%F_%T),$MODEL,${POLICY#"$TRAIN"/},$BEFORE-$((AFTER - 1)),\
$TAG" >> "$LOG"
fi
echo "saved $((AFTER - BEFORE)) corrections (exit $STATUS) to $ROOT"
[ -f "$LOG" ] && column -s, -t < "$LOG"
