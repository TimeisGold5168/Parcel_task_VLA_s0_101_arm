# benchmark (phase 06)

This folder records the benchmark runs on the robot and makes the
score sheets.

| File | Purpose |
|---|---|
| `eval_parcel.sh` | One entry point for all robot tests: a dataset for each model and level (or set), and a session log |
| `bench_sheets.py` | Frame sheets and a score template for one benchmark dataset |

## Run one benchmark set

1. Build the heap of the set.
2. Record the set:

   ```bash
   export CKPT_STEP=010000 FREEZE=1 RTC_EXACT_DELAY=1 COMPILE=default
   export ROLLOUT_ARGS="--inference.type=rtc \
   --inference.rtc.execution_horizon=10 --inference.queue_threshold=30"
   bash parcel_vla/benchmark/eval_parcel.sh smolvla_b_fb --set kind2 \
     --level medium --episodes 5 --parcels 3 --time 180 --reset 60
   ```

3. Make the sheets:
   `python -m parcel_vla.benchmark.bench_sheets rollout_bench_parcel_smolvla_b_fb_kind2`.
4. Fill in the outcome columns of `runs/bench/<dataset>/scores.csv`.

`eval_parcel.sh --show` prints the session log of all eval datasets.
The B-spline models (`*_bsp*`) run through `run_bsp_parcel.py` and
write `runs/results_bsp_parcel.csv`.

A success is a heap with all parcels cleared, 0 wrong labels and 0
help. My score sheets and the column list are in
[`results/benchmark/`](../../results/benchmark).
