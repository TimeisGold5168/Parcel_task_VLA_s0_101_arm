#!/usr/bin/env bash
# Delete the last N episodes (default 1) from the newest rollout dataset
# matching a prefix, and clear the last N filled rows for that (model, spot)
# in results_spots.csv.
# Usage: bash outputs/portfolio/delete_last_episode.sh [prefix] [count]
#   Example: bash outputs/portfolio/delete_last_episode.sh rollout_act151_X1 1
#   Example: bash outputs/portfolio/delete_last_episode.sh   (uses newest rollout_*)
set -euo pipefail

PREFIX=${1:-rollout_}
COUNT=${2:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HOME/.cache/huggingface/lerobot/Low101
cd "$HERE/../.."

DS_PATH=$(ls -dt "$DATA"/"${PREFIX}"* 2>/dev/null | head -1)
[[ -z "$DS_PATH" ]] && { echo "no dataset matching ${PREFIX}*"; exit 1; }
DS=$(basename "$DS_PATH")

TOTAL=$(python -c "import json; print(json.load(open('$DS_PATH/meta/info.json'))['total_episodes'])")
if (( COUNT >= TOTAL )); then
  echo "$DS has $TOTAL episode(s); removing the whole dataset folder..."
  rm -rf "$DS_PATH"
else
  INDICES=$(python -c "print(list(range($TOTAL - $COUNT, $TOTAL)))" | tr -d ' ')
  echo "Deleting episode index(es) $INDICES from $DS (total=$TOTAL)..."
  [[ -f "$DS_PATH/camera_alignment.json" ]] && cp "$DS_PATH/camera_alignment.json" /tmp/cam_align_bak.json
  lerobot-edit-dataset \
    --repo_id="Low101/$DS" \
    --root="$DS_PATH" \
    --operation.type=delete_episodes \
    --operation.episode_indices="$INDICES" \
    --push_to_hub=false
  [[ -f /tmp/cam_align_bak.json ]] && mv /tmp/cam_align_bak.json "$DS_PATH/camera_alignment.json"
  rm -rf "${DS_PATH}_old"
fi

python - "$HERE/results_spots.csv" "$DS" "$COUNT" <<'EOF'
import csv, sys
sys.path.insert(0, "outputs/portfolio")
from review_rollouts import parse_rollout_name
csv_path, ds_name, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
model, spot = parse_rollout_name(ds_name)
rows = list(csv.DictReader(open(csv_path)))
fields = list(rows[0].keys())
filled = [r for r in rows if r["model"] == model and r["spot"] == spot and r["stage_reached"]]
for r in filled[-count:]:
    for k in fields:
        if k not in ("model", "checkpoint", "spot", "trial"):
            r[k] = ""
    print(f"Cleared CSV row: {model} {spot} trial {r['trial']}")
w = csv.DictWriter(open(csv_path, "w", newline=""), fieldnames=fields)
w.writeheader(); w.writerows(rows)
EOF
