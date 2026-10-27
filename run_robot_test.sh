#!/usr/bin/env bash
# Robot test helper.
# Usage: run_robot_test.sh <model> <condition> <episodes> [instruction]
#   act51       = ACT 100k on the 51 square episodes (second attempt)
#   act151      = ACT all 151 episodes, 50k fine-tune ("act" alias)
#   act_aug     = ACT fine-tuned 25k on eps 51-150 + image aug
#   diffusion   = Diffusion 100k on all 151 episodes
#   smolvla     = SmolVLA 100k on all 151 episodes
#   smolvla_aug = SmolVLA fine-tuned 25k on eps 51-150 + image aug
# Example: run_robot_test.sh act_aug X1 3
# ACT uses the checkpoint default (100-step chunks, no ensembling);
# ACT_ENSEMBLE=1 turns temporal ensembling back on
# POLICY_ARGS="--policy.x=y ..." adds policy overrides (config tests)
# DISPLAY_DATA=false turns off the rerun viewer (faster control loop)
# Recordings: ~/.cache/huggingface/lerobot/Low101/rollout_<model>_<cond>_*
set -euo pipefail

MODEL=${1:?model: act51, act151, act_aug, diffusion, smolvla, smolvla_aug}
COND=${2:?condition name, e.g. normal, distractors, lamp, lang_red}
EPISODES=${3:?number of episodes}
TASK=${4:-"pick green cube and place into the bin"}

TRAIN=/home/zhenting.lee/lerobot/lerobot/outputs/train
WRIST_DEV=${WRIST_DEV:-/dev/video4}
FRONT_DEV=${FRONT_DEV:-/dev/video6}
EXTRA=()
if [[ "${ACT_ENSEMBLE:-0}" == 1 && "$MODEL" == act* ]]; then
  EXTRA=(--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01)
fi

case "$MODEL" in
  act51)
    POLICY=$TRAIN/act_second_attempt/checkpoints/100000/pretrained_model
    ;;
  act|act151)
    MODEL=act151
    POLICY=$TRAIN/act_all_ft50k/checkpoints/050000/pretrained_model
    ;;
  act_aug|actaug)
    MODEL=act_aug
    POLICY=$TRAIN/act_xbal_aug_ft25k/checkpoints/025000/pretrained_model
    ;;
  diffusion)
    POLICY=$TRAIN/diffusion_all_100k/checkpoints/100000/pretrained_model
    ;;
  smolvla)
    POLICY=$TRAIN/smolvla_all_100k/checkpoints/100000/pretrained_model
    MAP='{"observation.images.front": "observation.images.camera1",'
    MAP+=' "observation.images.wrist": "observation.images.camera2"}'
    EXTRA=(--rename_map="$MAP")
    ;;
  smolvla_aug|smolvlaaug)
    MODEL=smolvla_aug
    POLICY=$TRAIN/smolvla_xbal_aug_ft25k/checkpoints/025000/pretrained_model
    MAP='{"observation.images.front": "observation.images.camera1",'
    MAP+=' "observation.images.wrist": "observation.images.camera2"}'
    EXTRA=(--rename_map="$MAP")
    ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
# extra --policy.* overrides, e.g. POLICY_ARGS="--policy.n_action_steps=8"
if [[ -n "${POLICY_ARGS:-}" ]]; then
  read -r -a _pargs <<< "$POLICY_ARGS"
  EXTRA+=("${_pargs[@]}")
fi
echo "policy overrides: ${EXTRA[*]:-none}"

CAMS="{wrist: {type: opencv, index_or_path: $WRIST_DEV, width: 640,"
CAMS+=" height: 480, fps: 30}, front: {type: opencv, index_or_path:"
CAMS+=" $FRONT_DEV, width: 640, height: 480, fps: 30}}"

echo "model=$MODEL condition=$COND episodes=$EPISODES"
echo "instruction: $TASK"
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="$POLICY" \
  "${EXTRA[@]}" \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61034620-if00 \
  --robot.id=my_follower_arm \
  --robot.cameras="$CAMS" \
  --dataset.repo_id="Low101/rollout_${MODEL}_${COND}" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --display_data="${DISPLAY_DATA:-true}" \
  --dataset.push_to_hub=false
