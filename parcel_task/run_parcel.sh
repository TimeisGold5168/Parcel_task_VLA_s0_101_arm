#!/usr/bin/env bash
# Real-robot test of a parcel policy (no leader arm).
#
#   bash outputs/portfolio/parcel_task/run_parcel.sh <model> [cond] [eps]
#
#   diffusion = Diffusion 80k on parcel-sort-a (DDIM 10 steps)
#   smolvla   = SmolVLA 20k on parcel-sort-a
#   smolvla80k = SmolVLA 80k on parcel-sort-a
#   diffusion_b, smolvla_b = Diffusion 80k, SmolVLA 80k on parcel-sort-b
#   smolvla_b_dg = smolvla_b + 15k steps on parcel-sort-b-dagger
#   smolvla_b_dgt = the same on parcel-sort-b-dagger-trim
#   smolvla_b_dgg = gentle fine-tune (lr 1e-5, B stats, B 2x, 6k steps)
#   smolvla_b_fe, smolvla_b_box, smolvla_b_fb = smolvla_b + 5k steps
#     with far-edge picks, box-stack demos, or both (queue_skill.sh)
#   smolvla_b_fbs = smolvla_b_fb 10k + 10k steps with the stacked-parcel
#     corrections (queue_skill.sh fbs)
#   smolvla_b_all = smolvla_b_fb 10k + 20k steps on the balanced
#     all-skill mix (queue_all.sh); CKPT_STEP=010000 passed the gate
#   smolvla_b_sk3 = smolvla_b + 10k steps with one skill text for each
#     episode (queue_skill3.sh). Needs SKILL=box|bag|top (the start
#     text), e.g. SKILL=box run_parcel.sh smolvla_b_sk3. During the run,
#     keys 1/2/3 change the text of the next plans (box/bag/top box),
#     0 = back to the start text. Key log: debug_logs/skill_<time>.csv
#   smolvla_b_sk3d = the same with texts that share no word
#     (make_skill3.py --variant distinct; texts in skill3d_split.json)
#   smolvla_b_sk5d = 5 skills with texts that share no word (also far
#     edge and on table: SKILL=far|push, keys 4/5; skill5d_split.json)
#   PLANNER=log|drive (sk3 models): box/bag planner on the robot
#     (planner/live_planner.py); log = calls only, the keys stay with
#     the operator; drive = the planner sets the text. PLAN_FROM=probe
#     (default, learned habit) or view (ranked top view); WRIST_SWITCH=1
#     lets the wrist check change the text one time. Log:
#     debug_logs/planner_<time>.csv and planner_<time>_img/
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
# SmolVLA speed and debug (env, see rollout_mixed.py):
#   DEBUG=1 LABEL=<name>  log to debug_logs/<name>_<time>.npz and print
#     one line per plan; compare with rollout_debug.py debug_logs/*.npz
#   FREEZE=1  no autograd in RTC guidance (same output, ~70 ms faster)
#   COMPILE=default  compiled denoise step (first plan 4-20 s)
#   RTC_EXACT_DELAY=1  RTC skips the actions really used in the plan
set -euo pipefail

MODEL=${1:?model: diffusion, smolvla(80k), diffusion_b, smolvla_b(_dg)}
COND=${2:-test}
EPISODES=${3:-1}
TASK="sort the parcel so the label faces down and push it off the table"
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$HERE/../../train
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
case "$MODEL" in
  smolvla_b_sk3) export SKILL_SPLIT=$TRAIN/skill3_split.json ;;
  smolvla_b_sk3d) export SKILL_SPLIT=$TRAIN/skill3d_split.json ;;
  smolvla_b_sk5d) export SKILL_SPLIT=$TRAIN/skill5d_split.json ;;
esac
if [ -n "${SKILL_SPLIT:-}" ]; then
  case "${SKILL:-}" in
    box|bag|top|far|push) ;;
    *) echo "$MODEL needs SKILL=box|bag|top (sk5d: also far|push)"
       exit 1 ;;
  esac
  TASK=$("$PY" -c "import json, sys
print(json.load(open(sys.argv[1]))['texts'][sys.argv[2]])" \
    "$SKILL_SPLIT" "$SKILL")
  mkdir -p "$HERE/debug_logs"
  export LIVE_SKILL=1
  STAMP=$(date +%Y%m%d_%H%M%S)
  export SKILL_LOG=${SKILL_LOG:-$HERE/debug_logs/skill_$STAMP.csv}
  echo "$(date +%s.%3N),launch $SKILL" >> "$SKILL_LOG"
  if [ -n "${PLANNER:-}" ]; then
    export PLANNER_LOG=${PLANNER_LOG:-$HERE/debug_logs/planner_$STAMP.csv}
  fi
fi

EXTRA=()
case "$MODEL" in
  diffusion|diffusion_b)
    RUN=diffusion_parcel_a1_80k
    [ "$MODEL" = diffusion_b ] && RUN=diffusion_parcel_b_80k
    POLICY=$TRAIN/$RUN/checkpoints/last/pretrained_model
    # DDPM 100 steps = 0.8 s per plan; DDIM 10 steps = about 0.1 s
    EXTRA=(--policy.noise_scheduler_type=DDIM
           --policy.num_inference_steps="${STEPS:-10}")
    ;;
  smolvla|smolvla80k|smolvla_b|smolvla_b_dg*|smolvla_b_fe|smolvla_b_box|\
  smolvla_b_fb|smolvla_b_fe15|smolvla_b_fbs|smolvla_b_all|smolvla_b_sk3*)
    RUN=smolvla_parcel_a1_20k
    [ "$MODEL" = smolvla80k ] && RUN=smolvla_parcel_a1_80k
    [ "$MODEL" = smolvla_b ] && RUN=smolvla_parcel_b_80k
    [ "$MODEL" = smolvla_b_dg ] && RUN=smolvla_parcel_b_dagger_15k
    [ "$MODEL" = smolvla_b_dgt ] && RUN=smolvla_parcel_b_dagger_trim_15k
    [ "$MODEL" = smolvla_b_dgg ] && RUN=smolvla_parcel_b_dagger_gentle_6k
    case "$MODEL" in
      smolvla_b_fe|smolvla_b_box)
        RUN=smolvla_parcel_b_${MODEL#smolvla_b_}_5k ;;
      smolvla_b_fb) RUN=smolvla_parcel_b_fb_15k ;;
      smolvla_b_fe15) RUN=smolvla_parcel_b_fe_15k ;;
      smolvla_b_fbs) RUN=smolvla_parcel_b_fbs_10k ;;
      smolvla_b_all) RUN=smolvla_parcel_b_all_20k ;;
      smolvla_b_sk3|smolvla_b_sk3d|smolvla_b_sk5d)
        RUN=smolvla_parcel_b_${MODEL#smolvla_b_}_10k ;;
    esac
    # CKPT_STEP=005000 runs that checkpoint instead of the last one
    POLICY=$TRAIN/$RUN/checkpoints/${CKPT_STEP:-last}/pretrained_model
    MAP='{"observation.images.top": "observation.images.camera1",'
    MAP+=' "observation.images.wrist": "observation.images.camera2",'
    MAP+=' "observation.images.side": "observation.images.camera3"}'
    EXTRA=(--rename_map="$MAP")
    # moving average over 7 actions of each chunk (jitter); SMOOTH=0 = off
    export SMOOTH=${SMOOTH:-7}
    ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
if [ ! -d "$POLICY" ]; then
  echo "no checkpoint: $POLICY"
  echo "available (CKPT_STEP=...):" \
    $(ls "$(dirname "$(dirname "$POLICY")")" 2>/dev/null)
  exit 1
fi
export PREVIEW_LABEL="$MODEL $COND"
if [ "${DEBUG:-0}" != 0 ]; then
  mkdir -p "$HERE/debug_logs"
  LABEL=${LABEL:-$MODEL}
  export ROLLOUT_DEBUG=$HERE/debug_logs/${LABEL}_$(date +%Y%m%d_%H%M%S).npz
  export ROLLOUT_DEBUG_LABEL=$LABEL
  echo "debug: $ROLLOUT_DEBUG (args: ${ROLLOUT_ARGS:-none}," \
    "smooth=${SMOOTH:-7} freeze=${FREEZE:-0} compile=${COMPILE:-off}" \
    "exact_delay=${RTC_EXACT_DELAY:-0})"
fi
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
# a run stopped before its first episode leaves meta/info.json only
if [ "$RESUME" = false ] && [ -d "$ROOT" ] \
    && [ ! -d "$ROOT/data" ] && [ ! -d "$ROOT/videos" ]; then
  rm -rf "$ROOT"
fi

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
