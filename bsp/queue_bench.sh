#!/usr/bin/env bash
# Item 14 benchmark queue (offline, no robot motion): waits for the GPU,
# then records the reference and measures items 10-12, one change at a
# time. Results: bsp/bench_results/{plan,control}.jsonl, summary.md.
#   systemd-inhibit --what=sleep:idle:handle-lid-switch --why=bench \
#     nohup bash outputs/portfolio/bsp/queue_bench.sh \
#     > outputs/portfolio/bsp/bench_results/queue.out 2>&1 &
# STAGE=plan | control runs one part only (default: both).
set -uo pipefail

cd "$(dirname "$0")/../../.."
PY=/home/zhenting.lee/miniforge3/envs/lerobot/bin/python
B=outputs/portfolio/bsp
R=$B/bench_results
mkdir -p "$R"
PLAN=$R/plan.jsonl
CTRL=$R/control.jsonl
STAGE=${STAGE:-all}
note() { echo "$(date '+%F %T') $*"; }

BUSY="lerobot-train|queue_skill.sh|start_test.py|run_bsp_parcel.py"
BUSY+="|rollout_mixed.py|record_preview.py|lerobot-rollout"
while pgrep -f "$BUSY" >/dev/null; do
  sleep 60
done
note "GPU free"
git -C . log -1 --format='commit %H' > "$R/commit.txt" 2>/dev/null

bp() {  # bench_plan with a label
  local label=$1
  shift
  note "plan: $label"
  "$PY" $B/bench_plan.py --repeats 40 --warmup 5 --json "$PLAN" \
    --label "$label" "$@" 2>&1 | grep -v -i warn | tail -4
}

if [ "$STAGE" != control ]; then
  # reference: current code, the flags of the robot runs
  bp ref_diffusion --model diffusion_bsp_b --save-ref
  bp floor_diffusion --model diffusion_bsp_b
  bp ref_smolvla --model smolvla_bsp_b --save-ref
  bp floor_smolvla --model smolvla_bsp_b
  bp ref_smolvla_compile --model smolvla_bsp_b --mode static_cache \
    --compile-mode default

  # 10a: Diffusion engines, fp32 first (same precision), then fp16
  for dt in fp32 fp16; do
    note "build diffusion_bsp_b $dt"
    "$PY" $B/trt_plan.py build diffusion_bsp_b --dtype $dt 2>&1 \
      | grep -E "built|exported|Error" | tail -4
    bp "10a_diffusion_trt_$dt" --model diffusion_bsp_b --mode trt \
      --trt-dtype $dt
  done

  # 10b: SmolVLA engines (bf16, as the VLM now), alone and with the
  # compiled denoise steps of the robot runs
  note "build smolvla_bsp_b bf16"
  "$PY" $B/trt_plan.py build smolvla_bsp_b --dtype bf16 2>&1 \
    | grep -E "built|exported|Error" | tail -4
  bp 10b_smolvla_trt --model smolvla_bsp_b --mode trt
  bp 10b_smolvla_trt_compile --model smolvla_bsp_b --mode trt_static \
    --compile-mode default
fi

if [ "$STAGE" != plan ]; then
  # 11 / 12: tick timing while the planner runs (fake arm, no bus).
  # Diffusion-BSP at 1.5x: it plans most often.
  TRT=off
  [ -e $B/engines/diffusion_bsp_b_unet_fp16.engine ] && TRT=fp16
  lp() {
    local label=$1
    shift
    note "loop: $label"
    "$PY" $B/bench_control.py loop --seconds 40 --json "$CTRL" \
      --label "$label" "$@" 2>&1 | grep '^{' | tail -1
  }
  lp ref_python --control python
  lp floor_python --control python
  lp ref_python_video_cams --control python --video on --cams real
  lp 11_cpp --control cpp
  lp 11_cpp_video_cams --control cpp --video on --cams real
  lp "10a+11_cpp_trt_video_cams" --control cpp --video on --cams real \
    --trt $TRT
  lp "10a_python_trt_video_cams" --control python --video on \
    --cams real --trt $TRT
fi

"$PY" $B/bench_summary.py > "$R/summary.md"
note "done: $R/summary.md"
