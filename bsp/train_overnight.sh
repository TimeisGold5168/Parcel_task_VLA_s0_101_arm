#!/usr/bin/env bash
# Overnight queue: ACT-BSP then Diffusion-BSP (warm start), one at a time.
# Usage: bash outputs/portfolio/bsp/train_overnight.sh
# Logs: outputs/train/<run>.log   Checkpoints every 5k steps.
# Expected: ACT 30k steps ~4.2 h, Diffusion 40k steps ~4 h.
set -uo pipefail

cd "$(dirname "$0")/../../.."
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
TRAIN=outputs/train
BSP=outputs/portfolio/bsp

if pgrep -f lerobot-rollout >/dev/null; then
  echo "A lerobot-rollout is still running; stop it first (GPU)."
  exit 1
fi

run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" | tee -a "$TRAIN/bsp_queue.log"
  "$PY" "$BSP/train_bsp.py" "$@" --out "$TRAIN/$name" \
    > "$TRAIN/$name.log" 2>&1
  echo "$(date '+%F %T') end $name (exit $?)" | tee -a "$TRAIN/bsp_queue.log"
}

run act_bsp_ws30k --policy act \
  --init "$TRAIN/act_all_ft50k/checkpoints/050000/pretrained_model" \
  --steps 30000 --batch 8 --workers 4
run diffusion_bsp_ws40k --policy diffusion \
  --init "$TRAIN/diffusion_all_100k/checkpoints/100000/pretrained_model" \
  --steps 40000 --batch 16 --workers 4
