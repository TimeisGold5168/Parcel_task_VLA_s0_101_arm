# results

This folder holds the result files that the project page and the
README show. The scripts do not write here: new runs go to `runs/`.
The method and every number are in
[`docs/writeup.md`](../docs/writeup.md).

| Folder | Content | Made by |
|---|---|---|
| [`benchmark/`](benchmark) | Score sheets of the robot benchmark sets (`scores.csv` for each set) | `benchmark/eval_parcel.sh`, `bench_sheets.py` |
| [`offline_eval_b/`](offline_eval_b) | Offline tests 1-3 of the dataset B models | `evaluation/offline_eval_b.py` |
| [`inference_improvement/`](inference_improvement) | Plan and control benchmarks, and the summary tables | `inference_improvement/queue_bench.sh` |
| [`robot/`](robot) | B-spline robot runs of the parcel task, one row per episode | `robot/run_bsp_parcel.py` |
| [`finetuning/`](finetuning) | Offline scores of the new-skill checkpoints (start test, fit) | `finetuning/pick_checkpoint.py` |
| [`skill_texts/`](skill_texts) | Language tests of the skill-text models | `skill_texts/lang_test.py` |
| [`planner/`](planner) | Planner tests: zero-shot VLM, probes, next-parcel kind | `planner/vlm_planner.py`, `probe.py`, `next_kind.py` |
| [`bsp/`](bsp) | Offline grid of 1,920 B-spline executor settings (cube task) | `bsp/grid_executor.py` |
| [`cube_warmup/`](cube_warmup) | Cube task: spot tests (distractor), ACT ensembling, B-spline speeds | `cube_warmup/review_rollouts.py`, `robot/run_bsp.py` |

Most robot tests have few trials and one heap for each model. They
give an indication, not a controlled comparison.
