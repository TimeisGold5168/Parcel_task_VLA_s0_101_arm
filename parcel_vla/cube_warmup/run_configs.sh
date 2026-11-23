#!/usr/bin/env bash
# Inference-config comparison: one spot, each model config in turn.
# Usage: run_configs.sh [episodes] [spot]   (default: 5 episodes, X1)
# Configs, policy overrides and results live in results_configs.csv
# (columns spot = config id, policy_args = --policy.* overrides).
# Let every episode run the full 30 s (do not press Right arrow during
# the episode): loop_hz = frames / 30 s is only valid for full episodes.
set -euo pipefail

EPISODES=${1:-5}
SPOT=${2:-X1}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
DATA=$LOW
FRONT_DEV=${FRONT_DEV:-/dev/video6}
LOG=$RUNS/camcheck/alignment_log.csv
export RESULTS_CSV=${RESULTS_CSV:-$RUNS/results_configs.csv}
ALIGN_JSON=$(mktemp /tmp/align_XXXX.json)
case "$SPOT" in
  SQ|SQR) REF_EP=${REF_EP:-0} ;;
  *) REF_EP=${REF_EP:-130} ;;
esac
mkdir -p "$RUNS/camcheck"

# one line per config: model|cond|description|policy_args
mapfile -t CONFIGS < <("$PY" - "$RESULTS_CSV" <<'EOF'
import csv, sys
seen = []
for r in csv.DictReader(open(sys.argv[1])):
    key = (r["model"], r["spot"])
    if key not in [s[:2] for s in seen]:
        seen.append((r["model"], r["spot"], r["config"], r["policy_args"]))
for s in seen:
    print("|".join(s))
EOF
)

done_count() {
  "$PY" - "$RESULTS_CSV" "$1" "$2" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(sum(1 for r in rows if r["model"] == sys.argv[2]
          and r["spot"] == sys.argv[3] and r["stage_reached"]))
EOF
}

check_camera() {
  while true; do
    echo "Measuring front camera against episode $REF_EP ..."
    set +e
    "$PY" -m parcel_vla.setup.camera_align --check --device "$FRONT_DEV" \
      --ref-episode "$REF_EP" --out "$ALIGN_JSON" >/dev/null 2>&1
    set -e
    "$PY" - "$ALIGN_JSON" <<'EOF'
import json, math, sys
a = json.load(open(sys.argv[1]))
if "rot_deg" in a:
    shift = math.hypot(a["dx_px"], a["dy_px"])
    status = "ALIGNED" if a["ok"] else "OFFSET"
    print(f"camera vs ep{a['ref_episode']}: rot {a['rot_deg']:+.2f} deg,"
          f" shift {shift:.1f} px -> {status}")
else:
    print(f"camera check failed: {a.get('error')}")
EOF
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
  ds=$(ls -dt "$DATA"/rollout_"${model}"_"${cond}"_* 2>/dev/null | head -1)
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

for RUN in "${CONFIGS[@]}"; do
  IFS='|' read -r MODEL COND DESC ARGS <<< "$RUN"
  DONE=$(done_count "$MODEL" "$COND")
  NEEDED=$((EPISODES - DONE))
  if (( NEEDED <= 0 )); then
    echo
    echo "=== $COND | $DESC: $DONE/$EPISODES already recorded ==="
    read -r -p "Press Enter to skip, or type r + Enter to run more: " ans
    [[ "$ans" == "r" ]] && NEEDED=$EPISODES || continue
  fi
  echo
  echo "=== $COND | $DESC | $NEEDED trials ($DONE/$EPISODES done) ==="
  echo "Fold the arm to the home pose and place the green cube on $SPOT."
  echo "Let each episode run the full 30 s (no Right arrow in the episode)."
  read -r -p "Press Enter to load $MODEL (or type s to skip): " ans
  if [[ "$ans" == "s" ]]; then
    echo "skipped $COND"
    continue
  fi
  check_camera
  POLICY_ARGS="$ARGS" bash "$HERE/run_robot_test.sh" "$MODEL" "$COND" \
    "$NEEDED" || echo "rollout for $COND ended with an error; continuing"
  save_alignment "$MODEL" "$COND"
  "$PY" -m parcel_vla.cube_warmup.review_rollouts \
    "rollout_${MODEL}_${COND}_" || echo "auto-scoring failed for $COND"
done

rm -f "$ALIGN_JSON"
echo
echo "Config comparison done: $RESULTS_CSV"
