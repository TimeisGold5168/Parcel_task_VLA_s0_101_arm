## Plan time (bench_plan.py, 5 fixed dataset B frames, fixed noise, 40 repeats)

| Label | Variant | p50 (ms) | p95 (ms) | Err (deg) | Err (norm.) | Device mem (MB) | Gain p50 |
|---|---|---|---|---|---|---|---|
| ref_diffusion | diffusion_bsp_b eager | 87.7 | 105.7 | 0.000 | 0.0e+00 | 635 | - |
| floor_diffusion | diffusion_bsp_b eager | 87.6 | 107.0 | 0.000 | 0.0e+00 | 635 | 1.00x |
| ref_smolvla | smolvla_bsp_b eager | 258.0 | 261.4 | 0.000 | 0.0e+00 | 1285 | - |
| floor_smolvla | smolvla_bsp_b eager | 258.8 | 261.7 | 0.000 | 0.0e+00 | 1285 | 1.00x |
| ref_smolvla_compile | smolvla_bsp_b static_cache (default) | 148.0 | 163.2 | 0.141 | 6.0e-03 | 2195 | - |
| 10a_diffusion_trt_fp32 | diffusion_bsp_b trt [fp32] | 35.0 | 59.4 | 0.064 | 6.8e-04 | 1111 | 2.51x |
| 10a_diffusion_trt_fp16 | diffusion_bsp_b trt [fp16] | 13.1 | 13.2 | 0.149 | 1.6e-03 | 863 | 6.69x |
| 10b_smolvla_trt | smolvla_bsp_b trt [bf16] | 225.8 | 234.6 | 2.330 | 3.3e-01 | 1883 | 1.14x |
| 10b_smolvla_trt_compile | smolvla_bsp_b trt_static (default) [bf16] | 164.9 | 175.3 | 2.329 | 3.3e-01 | 3759 | 1.56x |

## Servo bus read, Present_Position, 6 servos (ms)

| Driver | p50 | p95 | max | errors |
|---|---|---|---|---|
| python_lerobot | 1.295 | 1.714 | 5.457 | - |
| cpp_bsp_ctl | 1.006 | 1.24 | 1.406 | 0 |

## Cameras: read_latest() time and frame age (ms)

| Camera | read p50 | read max | age p50 | age p95 |
|---|---|---|---|---|
| top | 0.03 | 0.077 | 24.174 | 32.224 |
| wrist | 0.007 | 0.013 | 29.935 | 57.624 |
| side | 0.004 | 0.012 | 31.556 | 57.448 |

## Control tick period while the planner runs (50 Hz = 20 ms, Diffusion-BSP B at 1.5x, fake arm)

| Label | Control | Plan engine | Video | Cameras | Tick p50 | Tick p95 | Tick max | Late % | Plan p50 | Plan p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| ref_python | python | off | off | fake | 20.0 | 20.102 | 20.552 | 0.0 | 96.153 | 160.56 |
| floor_python | python | off | off | fake | 19.997 | 20.119 | 20.372 | 0.0 | 79.202 | 131.818 |
| ref_python_video_cams | python | off | on | real | 19.999 | 20.133 | 30.694 | 0.05 | 82.483 | 147.219 |
| 11_cpp | cpp | off | off | fake | 20.0 | 20.067 | 20.165 | 0.0 | 80.278 | 114.164 |
| 11_cpp_video_cams | cpp | off | on | real | 20.0 | 20.071 | 25.044 | 0.0 | 80.788 | 111.402 |
| 10a+11_cpp_trt_video_cams | cpp | fp16 | on | real | 20.0 | 20.071 | 20.168 | 0.0 | 15.57 | 33.033 |
| 10a_python_trt_video_cams | python | fp16 | on | real | 19.999 | 20.113 | 20.911 | 0.0 | 19.796 | 53.303 |
