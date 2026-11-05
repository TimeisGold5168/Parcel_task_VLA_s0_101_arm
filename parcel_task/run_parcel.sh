#!/usr/bin/env bash
# Real-robot test of a parcel policy (no leader arm).
#
#   bash outputs/portfolio/parcel_task/run_parcel.sh <model> [cond] [eps]
#
#   diffusion = Diffusion 80k on parcel-sort-a (DDIM 10 steps)
#   smolvla   = SmolVLA 20k on parcel-sort-a
# Default (RECORD=0): one attempt, no recording, preview window
#   (LeRobot base strategy). Example: run_parcel.sh smolvla
# RECORD=1: episodes saved to a dataset, preview window, keys below.
#   Example: RECORD=1 run_parcel.sh smolvla up 3
# The robot starts from its pose at launch: put it at home first.
# It goes back to that pose after each attempt or episode.
# Keys: Right arrow = end episode or reset early, Left arrow = redo,
# Esc = stop. A preview window shows the cameras and the control loop
# rate (green at 27 Hz or more); PREVIEW=0 turns it off. Recordings:
#   ~/.cache/huggingface/lerobot/Low101/rollout_parcel_<model>_<cond>
# ROLLOUT_ARGS="--policy.x=y ..." adds overrides, for example
#   ROLLOUT_ARGS="--policy.num_inference_steps=100
#     --policy.noise_scheduler_type=DDPM" (training sampler, slow).
set -euo pipefail

MODEL=${1:?model: diffusion, smolvla}
COND=${2:-test}
EPISODES=${3:-1}
TASK="sort the parcel so the label faces down and push it off the table"
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$HERE/../../train
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python

EXTRA=()
case "$MODEL" in
  diffusion)
    POLICY=$TRAIN/diffusion_parcel_a1_80k/checkpoints/last/pretrained_model
    # DDPM 100 steps = 0.8 s per plan; DDIM 10 steps = about 0.1 s
    EXTRA=(--policy.noise_scheduler_type=DDIM
           --policy.num_inference_steps="${STEPS:-10}")
    ;;
  smolvla)
    POLICY=$TRAIN/smolvla_parcel_a1_20k/checkpoints/last/pretrained_model
    MAP='{"observation.images.top": "observation.images.camera1",'
    MAP+=' "observation.images.wrist": "observation.images.camera2",'
    MAP+=' "observation.images.side": "observation.images.camera3"}'
    EXTRA=(--rename_map="$MAP")
    # moving average over 7 actions of each chunk (jitter); SMOOTH=0 = off
    export SMOOTH=${SMOOTH:-7}
    ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
export PREVIEW_LABEL="$MODEL $COND"
if [[ -n "${ROLLOUT_ARGS:-}" ]]; then
  read -r -a _args <<< "$ROLLOUT_ARGS"
  EXTRA+=("${_args[@]}")
fi

V=/dev/v4l/by-id
# RealSense: use the node with YUYV colour (same as record_parcel.sh)
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
FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034620-if00
for d in "$RS" "$WRIST" "$SIDE" "$FOLLOWER"; do
  [ -e "$d" ] || { echo "missing device: $d"; exit 1; }
done

CAMERAS="{ \
top: {type: opencv, index_or_path: $RS, width: 848, \
height: 480, fps: 30, fourcc: YUYV}, \
wrist: {type: opencv, index_or_path: $WRIST, width: 640, \
height: 480, fps: 30, fourcc: MJPG}, \
side: {type: opencv, index_or_path: $SIDE, width: 640, \
height: 480, fps: 30, fourcc: MJPG}}"

REPO=${REPO:-Low101/rollout_parcel_${MODEL}_${COND}}
ROOT=$HOME/.cache/huggingface/lerobot/$REPO
RESUME=false
[ -d "$ROOT/meta/episodes" ] && RESUME=true

COMMON=(
  --policy.path="$POLICY"
  "${EXTRA[@]}"
  --robot.type=so101_follower
  --robot.port="$FOLLOWER"
  --robot.id=my_follower_arm
  --robot.num_read_retries=10
  --robot.cameras="$CAMERAS"
)
echo "overrides: ${EXTRA[*]} smooth=${SMOOTH:-0}"

if [ "${RECORD:-0}" = 0 ]; then
  # LeRobot base strategy: no dataset, no video encoder. One run = one
  # attempt of EP_TIME s (Ctrl+C stops early); then the robot goes back
  # to its pose at launch. The mpv preview shows the cameras. Rerun
  # (DISPLAY_DATA=true) is off: its viewer here does not take the data,
  # rr.log blocks and the loop falls to 5 Hz.
  echo "model=$MODEL: one attempt, ${EP_TIME:-90} s, no recording"
  exec "$PY" "$HERE/rollout_mixed.py" \
    --strategy.type=base \
    "${COMMON[@]}" \
    --task="$TASK" \
    --duration="${EP_TIME:-90}" \
    --display_data="${DISPLAY_DATA:-false}" \
    --display_compressed_images="${DISPLAY_COMPRESSED:-true}"
fi

echo "model=$MODEL condition=$COND episodes=$EPISODES resume=$RESUME"
echo "dataset: $ROOT"
"$PY" "$HERE/rollout_mixed.py" \
  --strategy.type=episodic \
  "${COMMON[@]}" \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s="${EP_TIME:-90}" \
  --dataset.reset_time_s="${RESET_TIME:-30}" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_queue_maxsize=150 \
  --display_data="${DISPLAY_DATA:-false}" \
  --resume="$RESUME"
