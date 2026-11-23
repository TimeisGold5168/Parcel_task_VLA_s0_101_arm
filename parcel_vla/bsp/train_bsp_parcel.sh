#!/usr/bin/env bash
# B-spline (BSP) training on a parcel dataset (laptop GPU, one job at a
# time). Warm start from a trained Diffusion or SmolVLA checkpoint.
#
#   bash parcel_vla/bsp/train_bsp_parcel.sh diffusion [init]
#   bash parcel_vla/bsp/train_bsp_parcel.sh smolvla [init]
#
# init: checkpoint folder or Hub model id. Default: the last checkpoint
#   of diffusion_parcel_b_80k or smolvla_parcel_b_80k in TRAIN_DIR.
# Default runs (dataset B): diffusion_bsp_parcel_b_40k (40k steps,
#   batch 16, ~5 h) and smolvla_bsp_parcel_b_40k (40k steps, batch 12,
#   ~14 h). Dataset A1 used 40k (Diffusion) and 20k (SmolVLA) steps.
# Make the targets first (once per dataset):
#   python -m parcel_vla.bsp.make_targets --dataset parcel-sort-b \
#     --max-error 0.5
# Env: DATASET (a dataset name of HF_USER, default parcel-sort-b), RUN,
#   STEPS, BATCH, TRAIN_DIR (outputs/train), PYTHON.
# Log: $TRAIN_DIR/<run>.log
set -uo pipefail

source "$(dirname "$0")/../env.sh"
DATASET=${DATASET:-parcel-sort-b}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_gpu() {
  while pgrep -f "lerobot-train|train_mixed|train_bsp --policy" \
      >/dev/null; do
    sleep 60
  done
}

case "${1:-}" in
  diffusion)
    INIT=${2:-$TRAIN/diffusion_parcel_b_80k/checkpoints/last/pretrained_model}
    RUN=${RUN:-diffusion_bsp_parcel_b_40k}
    ARGS=(--batch "${BATCH:-16}") ;;
  smolvla)
    INIT=${2:-$TRAIN/smolvla_parcel_b_80k/checkpoints/last/pretrained_model}
    RUN=${RUN:-smolvla_bsp_parcel_b_40k}
    ARGS=(--batch "${BATCH:-12}") ;;
  *)
    echo "usage: $0 diffusion|smolvla [init]"; exit 1 ;;
esac

wait_gpu
mkdir -p "$TRAIN"
echo "$(date '+%F %T') start $RUN" | tee -a "$TRAIN/bsp_queue.log"
"$PY" -m parcel_vla.bsp.train_bsp --policy "$1" --dataset "$DATASET" \
  --init "$INIT" --steps "${STEPS:-40000}" "${ARGS[@]}" --workers 8 \
  --save-freq 10000 --out "$TRAIN/$RUN" > "$TRAIN/$RUN.log" 2>&1
echo "$(date '+%F %T') end $RUN (exit $?)" | tee -a "$TRAIN/bsp_queue.log"
