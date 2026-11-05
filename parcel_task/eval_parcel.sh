#!/usr/bin/env bash
# Record policy evaluation episodes by difficulty level.
#
#   bash outputs/portfolio/parcel_task/eval_parcel.sh <model> \
#     --level easy|medium|hard [--episodes N] [--parcels N] \
#     [--parcel KIND] [--where TEXT] [--label up|down|mixed] [--note TEXT]
#
#   model     diffusion | smolvla | smolvla80k (see run_parcel.sh) |
#             diffusion_bsp | smolvla_bsp (run_bsp_parcel.py: asks s/f
#             per episode, no dataset; --no-record and --reset do not
#             apply)
#   --level   easy:   2-3 parcels, near the flip-zone side of the stack
#                     zone, mid reach, not at the zone edge
#             medium: 3-4 parcels, anywhere inside the stack zone
#             hard:   more than 4 parcels, at the zone edge or far
#                     reach, or the black box
#   --episodes  episodes in this session (default: --parcels, or 3)
#   --parcels   parcels in the heap (log only)
#   --parcel    parcel kinds, e.g. white_bag, black_box (log only)
#   --where     placement, e.g. near_flip, edge, far (log only)
#   --label     label side of the parcels (log only, default mixed)
#   --note      free text (log only)
#   --time      seconds per episode or attempt (default 90)
#   --reset     reset seconds between episodes (default 30)
#   --speed     BSP models only: curve speed vs the demos (default 1;
#               2 = twice as fast, 0.7 = slower)
#   --blend     BSP models only: fade between segments (s, default 0)
#   --fixed-noise  BSP models only: one start noise per episode
#   --debug     BSP models only: ep<n>_debug.npz per episode (timing,
#               margin, seam jumps, arm lag); read with
#               bsp/diagnose_bsp.py <run folder>
#   --no-record practice: --episodes attempts, nothing saved, no log
#   --show      print the session log of all eval datasets and exit
#
# One dataset per model and level (later sessions add episodes):
#   ~/.cache/huggingface/lerobot/Low101/rollout_eval_parcel_<model>_<level>
# Each session adds one line to <dataset>/eval_log.csv: time, model,
# checkpoint, settings, episode range and the log-only fields.
# Keys: Right arrow = end episode or reset early, Left arrow = redo,
# Esc = stop. Robot at home before the start.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
LOW=$HOME/.cache/huggingface/lerobot/Low101

show() {
  for f in "$LOW"/rollout_eval_parcel_*/eval_log.csv; do
    [ -e "$f" ] || { echo "no eval datasets yet"; break; }
    echo "== $(basename "$(dirname "$f")")"
    column -s, -t < "$f"
  done
  if [ -f "$HERE/results_bsp_parcel.csv" ]; then
    echo "== BSP models (results_bsp_parcel.csv)"
    column -s, -t < "$HERE/results_bsp_parcel.csv"
  fi
}

[ "${1:-}" = --show ] && { show; exit 0; }
MODEL=${1:?usage: eval_parcel.sh <model> --level easy|medium|hard ...}
shift
LEVEL="" EPISODES="" PARCELS="" KIND="" WHERE="" LABEL=mixed NOTE=""
NO_RECORD=0 SPEED=1
BSP_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --blend) BSP_ARGS+=(--blend "$2"); shift 2 ;;
    --fixed-noise) BSP_ARGS+=(--fixed-noise); shift ;;
    --debug) BSP_ARGS+=(--debug); shift ;;
    --no-record) NO_RECORD=1; shift ;;
    --time) export EP_TIME=$2; shift 2 ;;
    --speed) SPEED=$2; shift 2 ;;
    --reset) export RESET_TIME=$2; shift 2 ;;
    --level) LEVEL=$2; shift 2 ;;
    --episodes) EPISODES=$2; shift 2 ;;
    --parcels) PARCELS=$2; shift 2 ;;
    --parcel) KIND=$2; shift 2 ;;
    --where) WHERE=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --note) NOTE=$2; shift 2 ;;
    --show) show; exit 0 ;;
    *) echo "unknown flag $1"; exit 1 ;;
  esac
done
case "$LEVEL" in
  easy|medium|hard) ;;
  *) echo "--level must be easy, medium or hard"; exit 1 ;;
esac
EPISODES=${EPISODES:-${PARCELS:-3}}

if [ "$MODEL" = diffusion_bsp ] || [ "$MODEL" = smolvla_bsp ]; then
  # own executor (no LeRobot dataset): results_bsp_parcel.csv + videos
  exec "$PY" "$HERE/run_bsp_parcel.py" --model "$MODEL" --level "$LEVEL" \
    --episodes "$EPISODES" --duration "${EP_TIME:-90}" --speed "$SPEED" \
    --parcels "$PARCELS" --parcel "$KIND" --where "$WHERE" \
    --label "$LABEL" --note "$NOTE" "${BSP_ARGS[@]}"
fi

if [ "$NO_RECORD" = 1 ]; then
  # LeRobot base strategy: one attempt per run, nothing saved
  echo "level=$LEVEL model=$MODEL attempts=$EPISODES (no recording)"
  for i in $(seq 1 "$EPISODES"); do
    read -rp "Attempt $i/$EPISODES: robot at home, heap built? Enter. "
    RECORD=0 bash "$HERE/run_parcel.sh" "$MODEL" "$LEVEL" || true
  done
  exit 0
fi

# lerobot-rollout needs dataset names that start with rollout_
REPO=Low101/rollout_eval_parcel_${MODEL}_${LEVEL}
ROOT=$LOW/rollout_eval_parcel_${MODEL}_${LEVEL}
count() {
  [ -f "$ROOT/meta/info.json" ] || { echo 0; return; }
  "$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))\
['total_episodes'])" "$ROOT/meta/info.json"
}
BEFORE=$(count)

echo "level=$LEVEL model=$MODEL episodes=$EPISODES parcels=${PARCELS:-?}" \
  "kind=${KIND:-?} where=${WHERE:-?} label=$LABEL" \
  "time=${EP_TIME:-90}s reset=${RESET_TIME:-30}s"
read -rp "Robot at home, heap built? Press Enter to start. "

STATUS=0
RECORD=1 REPO=$REPO \
  bash "$HERE/run_parcel.sh" "$MODEL" "$LEVEL" "$EPISODES" || STATUS=$?
AFTER=$(count)

case "$MODEL" in
  diffusion) CKPT=diffusion_parcel_a1_80k/last
             SET="DDIM ${STEPS:-10} steps" ;;
  smolvla)   CKPT=smolvla_parcel_a1_20k/last
             SET="smooth ${SMOOTH:-7}" ;;
  smolvla80k) CKPT=smolvla_parcel_a1_80k/last
             SET="smooth ${SMOOTH:-7}" ;;
esac
SET="$SET; ${EP_TIME:-90} s"
LOG=$ROOT/eval_log.csv
if [ "$AFTER" -gt "$BEFORE" ]; then
  [ -f "$LOG" ] || echo "time,model,checkpoint,settings,level,episodes,\
parcels,kind,where,label,note" > "$LOG"
  echo "$(date +%F_%T),$MODEL,$CKPT,$SET,$LEVEL,$BEFORE-$((AFTER - 1)),\
${PARCELS},${KIND//,/;},${WHERE//,/;},$LABEL,${NOTE//,/;}" >> "$LOG"
fi

echo
echo "saved $((AFTER - BEFORE)) episodes (exit $STATUS) to $ROOT"
echo "model of this dataset: $MODEL ($CKPT, $SET)"
[ -f "$LOG" ] && column -s, -t < "$LOG"
