#!/usr/bin/env bash
# BSP training on the parcel dataset A1 (laptop GPU, one job at a time).
# Usage:
#   bash outputs/portfolio/parcel_task/train_bsp_parcel.sh diffusion
#   bash outputs/portfolio/parcel_task/train_bsp_parcel.sh smolvla <init>
# diffusion: waits for diffusion_parcel_a1_80k to end, then warm starts
#   from its last checkpoint (40k steps, batch 16, AMP from the config).
# smolvla: <init> = SmolVLA parcel checkpoint folder or Hub model id
#   (for example Low101/smolvla_parcel_a1_20k), 20k steps, batch 12.
# Targets: bsp/targets_parcel-sort-a_e0.5.npz (make_targets.py).
# Logs: outputs/train/<run>.log
set -uo pipefail

cd "$(dirname "$0")/../../.."
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
TRAIN=outputs/train
BSP=outputs/portfolio/bsp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_gpu() {
  while pgrep -f "train_mixed.py|train_bsp.py --policy" >/dev/null; do
    sleep 60
  done
}

run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" | tee -a "$TRAIN/bsp_queue.log"
  "$PY" "$BSP/train_bsp.py" --dataset parcel-sort-a "$@" \
    --out "$TRAIN/$name" > "$TRAIN/$name.log" 2>&1
  echo "$(date '+%F %T') end $name (exit $?)" | tee -a "$TRAIN/bsp_queue.log"
}

case "${1:-}" in
  diffusion)
    wait_gpu
    INIT=$TRAIN/diffusion_parcel_a1_80k/checkpoints/last/pretrained_model
    [ -e "$INIT/model.safetensors" ] || { echo "no $INIT"; exit 1; }
    run diffusion_bsp_parcel_40k --policy diffusion --init "$INIT" \
      --steps 40000 --batch 16 --workers 8 --save-freq 10000
    ;;
  smolvla)
    [ -n "${2:-}" ] || { echo "give the SmolVLA init"; exit 1; }
    wait_gpu
    run smolvla_bsp_parcel_20k --policy smolvla --init "$2" \
      --steps 20000 --batch 12 --workers 8 --save-freq 5000
    ;;
  *)
    echo "usage: $0 diffusion | smolvla <init>"; exit 1 ;;
esac
