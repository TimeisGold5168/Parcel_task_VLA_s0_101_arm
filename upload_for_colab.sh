#!/usr/bin/env bash
# Upload what the Colab fine-tune needs (run once, locally):
#   1. dataset   Low101/third-act-attempt   (private, ~1.5 GB)
#   2. ACT model Low101/act_all_ft50k       (checkpoint 050000, private)
# Requires: hf auth login  (token with write access)
set -euo pipefail

cd "$(dirname "$0")/../.."
DATA=$HOME/.cache/huggingface/lerobot/Low101/third-act-attempt
CKPT=outputs/train/act_all_ft50k/checkpoints/050000/pretrained_model

hf auth whoami >/dev/null 2>&1 || { echo "run: hf auth login"; exit 1; }

echo "== uploading dataset (this can take a while) =="
python - "$DATA" <<'EOF'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Low101/third-act-attempt", root=sys.argv[1])
ds.push_to_hub(private=True)
print("dataset uploaded: Low101/third-act-attempt")
EOF

echo "== uploading ACT checkpoint =="
hf upload Low101/act_all_ft50k "$CKPT" . --repo-type model --private
echo "model uploaded: Low101/act_all_ft50k"
