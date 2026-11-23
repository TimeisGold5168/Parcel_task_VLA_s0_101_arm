#!/usr/bin/env bash
# Run one test spot for every model in turn, then auto-score it.
# Usage: run_spot.sh <SPOT> [episodes]
#   SPOT: SQ SQR X1 X2 X3 X4 X5 UPL UPM UPR (or <SPOT>DUO directly)
#   Default episodes per session: 10
# Order per spot:
#   act151 -> act_aug -> diffusion ->
#   smolvla -> smolvla D -> smolvla DUO ->
#   smolvla_aug -> smolvla_aug D -> smolvla_aug DUO
#   D   = distractor object placed beside the green cube
#   DUO = untrained/unseen object; prompts you for the task instruction
# Camera-offset experiment: run_spot.sh SQCAM1 / SQCAM2
#   act151, act_aug, diffusion, smolvla, smolvla_aug with the front
#   camera moved on purpose (CAM1 small, CAM2 large shift at the cube)
set -euo pipefail

SPOT=${1:?spot, e.g. SQ, X1, UPL, or X1DUO}
EPISODES=${2:-10}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
DATA=$LOW
FRONT_DEV=${FRONT_DEV:-/dev/video6}
LOG=$RUNS/camcheck/alignment_log.csv
CSV=$RUNS/results_spots.csv
ALIGN_JSON=$(mktemp /tmp/align_XXXX.json)

CAM_MIN=0
CAM_MAX=0
case "$SPOT" in
  *CAM1) CAM_MIN=8; CAM_MAX=15 ;;
  *CAM2) CAM_MIN=20; CAM_MAX=35 ;;
esac
BASE_SPOT=${SPOT%CAM[12]}
BASE_SPOT=${BASE_SPOT%DUO}
BASE_SPOT=${BASE_SPOT%D}
case "$BASE_SPOT" in
  SQ|SQR) REF_DEFAULT=0 ;;
  *) REF_DEFAULT=130 ;;
esac
REF_EP=${REF_EP:-$REF_DEFAULT}

mkdir -p "$RUNS/camcheck"
if [[ ! -f "$LOG" ]]; then
  echo "time,spot,model,dataset,ref_episode,rot_deg,dx_px,dy_px,ok" \
    > "$LOG"
fi

done_count() {
  "$PY" - "$CSV" "$1" "$2" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
n = sum(1 for r in rows if r["model"] == sys.argv[2]
        and r["spot"] == sys.argv[3] and r["stage_reached"])
print(n)
EOF
}

check_camera() {
  while true; do
    echo "Measuring front camera against episode $REF_EP ..."
    set +e
    "$PY" -m parcel_vla.setup.camera_align --check --device "$FRONT_DEV" \
      --ref-episode "$REF_EP" --out "$ALIGN_JSON" 2>/dev/null >/dev/null
    code=$?
    set -e
    "$PY" - "$ALIGN_JSON" "$CAM_MIN" "$CAM_MAX" <<'EOF'
import json, math, sys
a = json.load(open(sys.argv[1]))
lo, hi = float(sys.argv[2]), float(sys.argv[3])
if "rot_deg" in a:
    shift = math.hypot(a["dx_px"], a["dy_px"])
    if hi > 0:
        ok = lo <= shift <= hi
        status = (f"IN TARGET {lo:.0f}-{hi:.0f} px" if ok
                  else f"OUT OF TARGET {lo:.0f}-{hi:.0f} px - move it")
    else:
        status = "ALIGNED" if a["ok"] else "OFFSET"
    print(f"camera vs ep{a['ref_episode']}: rot {a['rot_deg']:+.2f} deg,"
          f" shift {shift:.1f} px -> {status}")
else:
    print(f"camera check failed: {a.get('error')}")
EOF
    if (( CAM_MAX > 0 )); then
      echo "Offset test: the shift must be $CAM_MIN-$CAM_MAX px."
      echo "In the window, move the camera until the shift is in range."
    fi
    read -r -p "Type a + Enter to realign, or Enter to continue: " k
    if [[ "$k" == "a" ]]; then
      "$PY" -m parcel_vla.setup.camera_align --device "$FRONT_DEV" \
        --ref-episode "$REF_EP"
    else
      return 0
    fi
  done
}

save_alignment() {
  local model=$1 cond=$2 ds
  ds=$(ls -dt "$DATA"/rollout_"${model}"_"${cond}"_* 2>/dev/null \
    | head -1 || true)
  [[ -z "$ds" ]] && return
  cp "$ALIGN_JSON" "$ds/camera_alignment.json"
  "$PY" - "$ALIGN_JSON" "$cond" "$model" "$(basename "$ds")" "$LOG" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
row = [a.get("time", ""), sys.argv[2], sys.argv[3], sys.argv[4],
       a.get("ref_episode", ""), a.get("rot_deg", ""),
       a.get("dx_px", ""), a.get("dy_px", ""), a.get("ok", "")]
open(sys.argv[5], "a").write(",".join(map(str, row)) + "\n")
EOF
}

if [[ "$SPOT" == *DUO ]]; then
  RUNS=("smolvla $SPOT" "smolvla_aug $SPOT")
elif (( CAM_MAX > 0 )); then
  RUNS=(
    "act151 $SPOT"
    "act_aug $SPOT"
    "diffusion $SPOT"
    "smolvla $SPOT"
    "smolvla_aug $SPOT"
  )
else
  RUNS=(
    "act151 $SPOT"
    "act_aug $SPOT"
    "diffusion $SPOT"
    "smolvla $SPOT"
    "smolvla ${SPOT}D"
    "smolvla ${SPOT}DUO"
    "smolvla_aug $SPOT"
    "smolvla_aug ${SPOT}D"
    "smolvla_aug ${SPOT}DUO"
  )
fi

DEFAULT_DUO_TASK="pick blue cube and place into the bin"

for RUN in "${RUNS[@]}"; do
  read -r MODEL COND <<< "$RUN"
  DONE=$(done_count "$MODEL" "$COND")
  NEEDED=$((EPISODES - DONE))
  if (( NEEDED <= 0 )); then
    echo
    echo "=== $COND | $MODEL: $DONE/$EPISODES already recorded ==="
    read -r -p "Press Enter to skip, or type r + Enter to run more: " ans
    [[ "$ans" == "r" ]] && NEEDED=$EPISODES || continue
  fi
  echo
  echo "=== $COND | model $MODEL | $NEEDED trials ($DONE/$EPISODES done) ==="
  echo "Fold the arm to the home pose and place the target on $BASE_SPOT."
  TASK="pick green cube and place into the bin"
  if [[ "$COND" == *DUO ]]; then
    echo "UNSEEN OBJECT (DUO): place your target/distractor object(s)."
    read -r -e -p "Instruction [$DEFAULT_DUO_TASK]: " user_task
    # drop escape codes from arrow keys; lerobot parses the task as YAML
    user_task=$(printf '%s' "$user_task" \
      | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tr -d '[:cntrl:]')
    TASK=${user_task:-$DEFAULT_DUO_TASK}
    echo "Task instruction: $TASK"
  elif [[ "$COND" == *D ]]; then
    echo "DISTRACTOR: also put a blue or red cube next to the green one."
  fi
  read -r -p "Press Enter to load $MODEL (or type s to skip): " ans
  if [[ "$ans" == "s" ]]; then
    echo "skipped $MODEL $COND"
    continue
  fi
  check_camera
  bash "$HERE/run_robot_test.sh" "$MODEL" "$COND" "$NEEDED" "$TASK" || \
    echo "rollout for $MODEL $COND ended with an error; continuing"
  save_alignment "$MODEL" "$COND"
  "$PY" -m parcel_vla.cube_warmup.review_rollouts \
    "rollout_${MODEL}_${COND}_" || echo "auto-scoring failed for $MODEL $COND"
  "$PY" -m parcel_vla.cube_warmup.vlm_frames "rollout_${MODEL}_${COND}_" \
    > /dev/null 2>&1 || echo "frame grids failed for $MODEL $COND"
done

rm -f "$ALIGN_JSON"
echo
echo "Spot $SPOT done. Review sheets: $RUNS/spot_review/"
echo "Frame grids for the vision judge: $RUNS/spot_review/vlm/"
echo "Camera log: $LOG"
if (( CAM_MAX > 0 )); then
  echo "Offset test done: realign the camera before other spots:"
  echo "  python -m parcel_vla.setup.camera_align --ref-episode $REF_EP"
fi
