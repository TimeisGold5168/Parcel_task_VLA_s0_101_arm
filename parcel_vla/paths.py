"""Shared paths, names and devices.

Each value can be set with the environment variable of the same name.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Training runs: <TRAIN>/<run>/checkpoints/<step>/pretrained_model.
# The default is relative to the current folder.
TRAIN = Path(os.environ.get("TRAIN_DIR", "outputs/train")).resolve()

# Robot runs, debug logs and new result tables (git-ignored).
RUNS = Path(os.environ.get("RUNS_DIR", REPO / "runs"))

# Hugging Face namespace of the datasets, and its local LeRobot cache.
HF_USER = os.environ.get("HF_USER", "Low101")
HF_LEROBOT_HOME = Path(os.environ.get(
    "HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
LOW = HF_LEROBOT_HOME / HF_USER

# Robot devices. The by-id names do not change between reboots.
_SERIAL = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_"
_V4L = "/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-"
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", _SERIAL + "5B61034620-if00")
LEADER_PORT = os.environ.get("LEADER_PORT", _SERIAL + "5B79016363-if00")
WRIST_CAM = os.environ.get("WRIST_CAM", _V4L + "2607053-video-index0")
SIDE_CAM = os.environ.get("SIDE_CAM", _V4L + "2607071-video-index0")
# Top camera: empty = the RealSense video node with YUYV colour.
TOP_CAM = os.environ.get("TOP_CAM", "")


def dataset_id(name):
    """Hub id of a dataset name, for example parcel-sort-b."""
    return name if "/" in name else f"{HF_USER}/{name}"
