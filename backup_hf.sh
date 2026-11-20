#!/usr/bin/env bash
# Back up the datasets and the key models to the Hugging Face Hub
# (private repos, namespace Low101).
#
# Usage:
#   hf auth login                       # once, token with write access
#   bash outputs/portfolio/backup_hf.sh datasets
#   bash outputs/portfolio/backup_hf.sh models
#
# hf upload only sends files that are not on the Hub yet, so you can
# run it again after a stop. Each dataset gets the tag v3.0 (LeRobot
# loads a dataset from the Hub by its codebase version tag).
set -u
HF=${HF:-$HOME/miniforge3/envs/lerobot/bin/hf}
NS=${NS:-Low101}
DATA=$HOME/.cache/huggingface/lerobot/Low101
TRAIN=$(cd "$(dirname "$0")/../train" && pwd)

DATASETS=(
  parcel-sort-a parcel-sort-b parcel-sort-b-fb parcel-sort-b-all
  parcel-sort-b-sk3 parcel-sort-b-sk3d parcel-sort-b-sk5d
  parcel-skill-boxstack parcel-skill-boxstack-cut parcel-approach-demo
  second-act-attempt third-act-attempt
  lerobot-so101-first-act_20261015_205345
  lerobot-so101-learning_20261015_201504
  lerobot-so101-learning_20261015_201950
  lerobot-so101-learning_20261015_202357
)
MODELS=(
  diffusion_bsp_parcel_b_40k smolvla_parcel_b_80k smolvla_bsp_parcel_b_40k
  diffusion_parcel_b_80k smolvla_parcel_b_fb_15k smolvla_parcel_b_sk3d_10k
  smolvla_parcel_b_sk5d_10k
)

failed=()
case "${1:-}" in
  datasets)
    for d in "${DATASETS[@]}"; do
      echo "== dataset $NS/$d"
      if "$HF" upload "$NS/$d" "$DATA/$d" . --repo-type dataset \
          --private --commit-message "Back up $d"; then
        "$HF" repos tag create "$NS/$d" v3.0 --repo-type dataset \
          2>/dev/null || echo "   tag v3.0 is already there"
      else
        failed+=("$d")
      fi
    done ;;
  models)
    for m in "${MODELS[@]}"; do
      src=$TRAIN/$m/checkpoints/last/pretrained_model
      echo "== model $NS/$m"
      "$HF" upload "$NS/$m" "$src" . --repo-type model --private \
        --commit-message "Back up $m (last checkpoint)" || failed+=("$m")
    done ;;
  *)
    echo "usage: $0 datasets|models"; exit 2 ;;
esac

if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED: ${failed[*]} (run the same command again)"; exit 1
fi
echo "done"
