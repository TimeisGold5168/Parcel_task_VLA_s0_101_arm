#!/usr/bin/env bash
# Laptop queue for dataset A1, one GPU job at a time:
#   1. wait for Diffusion (diffusion_parcel_a1_80k) to end
#   2. Diffusion-BSP 40k (train_bsp_parcel.sh diffusion)
#   3. SmolVLA 20k (train_smolvla_a1.sh)
# Usage: nohup bash outputs/portfolio/parcel_task/queue_a1.sh \
#          > outputs/train/queue_a1.out 2>&1 &
# Progress: outputs/train/queue_a1.log
set -uo pipefail

cd "$(dirname "$0")/../../.."
HERE=outputs/portfolio/parcel_task
LOG=outputs/train/queue_a1.log

note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

note "queue start: waiting for Diffusion"
while pgrep -f "train_mixed.py" >/dev/null; do sleep 60; done
note "Diffusion ended"

bash "$HERE/train_bsp_parcel.sh" diffusion
note "Diffusion-BSP ended"

note "start SmolVLA"
bash "$HERE/train_smolvla_a1.sh" > outputs/train/smolvla_parcel_a1_20k.log 2>&1
note "SmolVLA ended (exit $?)"
