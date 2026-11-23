#!/usr/bin/env bash
# Upload what the Colab fine-tune needs (run once, locally):
#   1. dataset   $HF_USER/third-act-attempt   (private, ~1.5 GB)
#   2. ACT model $HF_USER/act_all_ft50k       (checkpoint 050000, private)
# Requires: hf auth login  (token with write access)
set -euo pipefail

source "$(dirname "$0")/../env.sh"
DATA=$LOW/third-act-attempt
CKPT=$TRAIN/act_all_ft50k/checkpoints/050000/pretrained_model

hf auth whoami >/dev/null 2>&1 || { echo "run: hf auth login"; exit 1; }

echo "== uploading dataset (this can take a while) =="
"$PY" - "$DATA" "$HF_USER/third-act-attempt" <<'EOF'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[2], root=sys.argv[1])
ds.push_to_hub(private=True)
print(f"dataset uploaded: {sys.argv[2]}")
EOF

echo "== uploading ACT checkpoint =="
hf upload "$HF_USER/act_all_ft50k" "$CKPT" . --repo-type model --private
echo "model uploaded: $HF_USER/act_all_ft50k"
