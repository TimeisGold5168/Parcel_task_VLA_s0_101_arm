# docs

| Document | Content |
|---|---|
| [`writeup.md`](writeup.md) | The full technical record: cube task and parcel task, every test, number and limit |
| [`test_protocol.md`](test_protocol.md) | Cube task: the robot test protocol, 10 taped spots with 2 trials each |
| [`spot_test_plan.md`](spot_test_plan.md) | Cube task: the spot test of the 151-episode models |
| [`patches/`](patches) | The change to the LeRobot source (base commit `64b23178`) |
| [`baseline_pre10/`](baseline_pre10) | Code, packages, GPU and commit before the speed items 10-12: the fixed reference of the benchmarks |

The write-up keeps the file names of the time of each test. The
[main README](../README.md#repository-layout) shows the new place of
each script. Raw logs and some early result tables are not in this
repository.

The cube results that the project page shows are in
[`results/cube_warmup/`](../results/cube_warmup):

| File | Content |
|---|---|
| `results_spots.csv` | Spot-test results, with the distractor test |
| `results_act_ensemble.csv` | ACT runs with temporal ensembling |
| `results_bsp.csv` | ACT-BSP and Diffusion-BSP speed runs |
