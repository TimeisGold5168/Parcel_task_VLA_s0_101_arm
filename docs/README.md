# Documents: cube task

This folder holds the test protocols and plans of the cube task. The
cube task is the warm-up before the parcel task. The results are in
[`../writeup.md`](../writeup.md), and the CSV files are at the
repository root.

- [`test_protocol.md`](test_protocol.md): the robot test protocol,
  10 taped spots with 2 trials each.
- [`spot_test_plan.md`](spot_test_plan.md): the spot test of the
  151-episode models.
- [`robot_tests_language_robustness.md`](robot_tests_language_robustness.md):
  the language and robustness tests on the robot.

Result files at the repository root:

| File | Content |
|---|---|
| `results.csv` | Robot results of `test_protocol.md` |
| `results_spots.csv` | Spot-test results |
| `results_language_robustness.csv` | Language and robustness tests |
| `results_configs.csv` | Inference-config comparison |
| `results_act_ensemble.csv` | ACT runs with temporal ensembling |
| `results_bsp.csv` | ACT-BSP and Diffusion-BSP speed runs |

The commands in these documents start from the LeRobot root, for
example `bash outputs/portfolio/run_spot.sh SQ`.
