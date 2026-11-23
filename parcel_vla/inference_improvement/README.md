# inference_improvement (phase 03)

This folder makes each plan arrive in time and keeps the 50 Hz control
loop steady. Every change was measured against one fixed reference.
The results are in
[`results/inference_improvement/summary.md`](../../results/inference_improvement/summary.md).

| File | Purpose |
|---|---|
| `fast_denoise.py` | SmolVLA: read-only prefix K/V cache + `torch.compile` of the denoising step |
| `trt_plan.py` | TensorRT engines for Diffusion-BSP and SmolVLA-BSP |
| `cpp_control.py` | Python side of the C++ control process `bsp_ctl` |
| `ctl/` | `bsp_ctl.cpp` (50 Hz loop and servo bus in C++), `Makefile`, `test_ctl.py` |
| `bench_plan.py` | Plan time and output error against a saved reference |
| `bench_control.py` | Servo bus, camera and control-tick benchmarks (no motion) |
| `bench_summary.py` | Markdown tables from the benchmark results |
| `queue_bench.sh` | The full benchmark queue, one change at a time |

The TensorRT engines (`engines/`) and the benchmark references
(`bench_plan_ref_*.pt`) are made next to the scripts. Git ignores them.
New benchmark results go to `runs/bench_results/`.

## Results

| Change | Before | After |
|---|---|---|
| Diffusion-BSP, TensorRT FP16 (offline p50) | 87.7 ms | 13.1 ms, error 0.149° |
| Diffusion-BSP, TensorRT FP16 (robot) | 87 ms | 18 ms |
| SmolVLA denoising step, `fast_denoise.py` | 13.2 ms | 3.4 ms, max difference 0.106° |
| SmolVLA-BSP plan, compiled | 258 ms | 148 ms |
| SmolVLA-BSP, TensorRT BF16 | 258 ms | 226 ms, error 2.33°: not used |
| Control tick max, C++ process + TensorRT | 30.7 ms | 20.2 ms |
| Servo bus read max, Python → C++ | 5.46 ms | 1.41 ms |

A SmolVLA plan keeps a fixed 94 ms for vision and the language prefix.
That part is the floor with this GPU.

## Commands

```bash
# TensorRT engines of one model, then run it on the robot with them
python -m parcel_vla.inference_improvement.trt_plan build diffusion_bsp_b --dtype fp16
python -m parcel_vla.robot.run_bsp_parcel --model diffusion_bsp_b --level easy \
  --speed 1.5 --trt fp16

# C++ control process: build, test offline, then use --control cpp
make -C parcel_vla/inference_improvement/ctl
python -m parcel_vla.inference_improvement.ctl.test_ctl

# plan benchmark: save the reference, then measure a change
python -m parcel_vla.inference_improvement.bench_plan --model diffusion_bsp_b --save-ref
python -m parcel_vla.inference_improvement.bench_plan --model diffusion_bsp_b \
  --mode trt --trt-dtype fp16

# tables from my published results
python -m parcel_vla.inference_improvement.bench_summary results/inference_improvement
```

The RTC fixes for SmolVLA (`FREEZE`, `RTC_EXACT_DELAY`, `COMPILE`) are
in [`../robot/rollout_mixed.py`](../robot/rollout_mixed.py).
