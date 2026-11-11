#!/usr/bin/env bash
# Record one heap block of the parcel task with 3 cameras.
#
#   bash outputs/portfolio/parcel_task/record_parcel.sh <block> [episodes]
#
# Block 1 makes the dataset; later blocks add to it (--resume=true).
# A large preview window shows all 3 cameras (PREVIEW=0 turns it off;
# PREVIEW_W sets its width, default 1600).
# Keys: Right arrow = end episode early, Left arrow = re-record the
# episode, Esc = stop and save. Ctrl+C loses the current episode.
# Space = phase mark (needs the preview, PREVIEW=1). With the preview,
# keys work only while the preview window has the focus.
# Episode flow: home > push > Space > flip > Space > exit push > home.
# If the label is down: home > push > Space > exit push > home.
# STAGE=b: dataset B (Low101/parcel-sort-b, schedule_B.csv), grasp
# approach: ready pose > grasp > pull (+ flip if label up) > release
# label down in the flip zone > Space > exit push > ready pose.
set -euo pipefail

BLOCK=${1:?usage: record_parcel.sh <block> [episodes]}
STAGE=${STAGE:-full}
SCHED=$(dirname "$0")/schedule_A1.csv
if [ "$STAGE" = b ]; then
  SCHED=$(dirname "$0")/schedule_B.csv
  REPO=${REPO:-Low101/parcel-sort-b}
  EP_TIME=${EP_TIME:-60}
fi
HEAP_N=$(awk -F, -v b="$BLOCK" 'NR > 1 && $1 == b' "$SCHED" | wc -l)
N_HEAPS=$(awk -F, 'NR > 1 { print $1 }' "$SCHED" | sort -n | tail -1)
[ "$HEAP_N" -gt 0 ] || { echo "heap $BLOCK not in $SCHED"; exit 1; }
# Default: one episode per parcel in this heap.
EPISODES=${2:-$HEAP_N}
REPO=${REPO:-Low101/parcel-sort-a}
ROOT=${ROOT:-$HOME/.cache/huggingface/lerobot/$REPO}
# 848x480 top view: same 16:9 field of view as 1280x720 at less than
# half the pixels, so the AV1 encoder keeps up (1280x720 dropped
# frames under load). 640x480 cuts the sides of the view. Needs the
# RealSense on USB 3 (monitor hub or Thunderbolt port).
TOP_W=${TOP_W:-848}
TOP_H=${TOP_H:-480}
TASK="sort the parcel so the label faces down and push it off the table"

V=/dev/v4l/by-id
# The RealSense depth, IR and colour nodes share one by-id name, so the
# by-id link can point to any of them; use the node with YUYV colour.
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
top: {type: opencv, index_or_path: $RS, width: $TOP_W, \
height: $TOP_H, fps: 30, fourcc: YUYV}, \
wrist: {type: opencv, index_or_path: $WRIST, width: 640, \
height: 480, fps: 30, fourcc: MJPG}, \
side: {type: opencv, index_or_path: $SIDE, width: 640, \
height: 480, fps: 30, fourcc: MJPG}}"

RESUME=false
FIRST_EP=0
if [ -d "$ROOT/meta/episodes" ]; then
  RESUME=true
  FIRST_EP=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))\
['total_episodes'])" "$ROOT/meta/info.json")
fi
if [ "$RESUME" = false ] && [ -d "$ROOT" ]; then
  echo "$ROOT exists but has no saved episodes; remove it first:"
  echo "  rm -rf $ROOT"
  exit 1
fi

echo "heap $BLOCK of $N_HEAPS: $EPISODES episodes -> $ROOT" \
  "(resume=$RESUME)"
awk -F, -v b="$BLOCK" '
  { sub(/\r$/, "") }
  NR > 1 && $1 == b { k = $3 " " $4; n[k]++; if ($5 == "up") u[k]++ }
  END { print "  Build the heap with:"
        for (k in n) printf "    %d %s  (%d label up, %d label down)\n",
                            n[k], k, u[k], n[k] - u[k] }' "$SCHED"
if [ "$STAGE" = b ]; then
  echo "  Start and end each episode at the same ready pose."
  echo "  Space when the parcel is released label down in the flip zone."
else
  echo "  Space at the end of push, and at the end of flip (label up only)."
fi
echo "  Click the preview window first: keys work only there."
read -rp "Heap built as listed? Press Enter to start. "

RECORD=(python "$(dirname "$0")/record_preview.py")
[ "${PREVIEW:-1}" = 0 ] && RECORD=(lerobot-record)

"${RECORD[@]}" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER" \
  --robot.id=my_follower_arm \
  --robot.num_read_retries=10 \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER" \
  --teleop.id=my_leader_arm \
  --teleop.num_read_retries=10 \
  --display_data=false \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s="${EP_TIME:-120}" \
  --dataset.reset_time_s="${RESET_TIME:-10}" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_queue_maxsize=150 \
  --resume="$RESUME"

echo "$(date +%F_%T) block $BLOCK episodes $EPISODES" \
  >> "$(dirname "$0")/record_log.txt"

CHECK=()
[ "$STAGE" = b ] && CHECK=(--home median)
python "$(dirname "$0")/check_episodes.py" --root "$ROOT" --from "$FIRST_EP" \
  "${CHECK[@]}"
