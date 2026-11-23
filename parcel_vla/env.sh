# Shared settings of the shell scripts (source this file). Each value
# can be set with the environment variable of the same name, as in
# parcel_vla/paths.py.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
LEROBOT_TRAIN=${LEROBOT_TRAIN:-lerobot-train}
# Training runs; the default is relative to the current folder.
TRAIN=${TRAIN_DIR:-outputs/train}
# Robot runs, debug logs and new result tables (git-ignored).
RUNS=${RUNS_DIR:-$REPO_DIR/runs}
# Hugging Face namespace of the datasets, and its local LeRobot cache.
HF_USER=${HF_USER:-Low101}
LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
LOW=$LEROBOT_HOME/$HF_USER

# Split file of a skill-text model: the copy in TRAIN (made by
# make_skill3.py), else the copy in this repository.
skill_split() {
  if [ -f "$TRAIN/$1" ]; then
    echo "$TRAIN/$1"
  else
    echo "$REPO_DIR/parcel_vla/skill_texts/splits/$1"
  fi
}

# Robot devices. The by-id names do not change between reboots.
_SERIAL=/dev/serial/by-id/usb-1a86_USB_Single_Serial_
_V4L=/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-
FOLLOWER_PORT=${FOLLOWER_PORT:-${_SERIAL}5B61034620-if00}
LEADER_PORT=${LEADER_PORT:-${_SERIAL}5B79016363-if00}
WRIST_CAM=${WRIST_CAM:-${_V4L}2607053-video-index0}
SIDE_CAM=${SIDE_CAM:-${_V4L}2607071-video-index0}

# Top camera: TOP_CAM, or the RealSense video node with YUYV colour.
# The RealSense depth, IR and colour nodes share one by-id name, so
# the by-id link can point to any of them.
top_cam() {
  if [ -n "${TOP_CAM:-}" ]; then
    echo "$TOP_CAM"
    return
  fi
  local d
  for d in /sys/class/video4linux/video*; do
    grep -q RealSense "$d/name" 2>/dev/null || continue
    if v4l2-ctl -d "/dev/${d##*/}" --list-formats 2>/dev/null \
        | grep -q YUYV; then
      echo "/dev/${d##*/}"
      return
    fi
  done
}

# The camera config of the parcel task: top (848x480 YUYV), wrist and
# side (640x480 MJPG), all at 30 fps. $1, $2 = top camera width and
# height (848, 480).
parcel_cameras() {
  local top
  top=$(top_cam)
  echo "{ \
top: {type: opencv, index_or_path: $top, width: ${1:-848}, \
height: ${2:-480}, fps: 30, fourcc: YUYV}, \
wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, \
height: 480, fps: 30, fourcc: MJPG}, \
side: {type: opencv, index_or_path: $SIDE_CAM, width: 640, \
height: 480, fps: 30, fourcc: MJPG}}"
}

# Stop if a device is missing.
need_devices() {
  local d
  for d in "$@"; do
    if [ -z "$d" ]; then
      echo "missing device: no RealSense colour node (set TOP_CAM)"
      exit 1
    fi
    [ -e "$d" ] || { echo "missing device: $d (see parcel_vla/env.sh)"
                     exit 1; }
  done
}
