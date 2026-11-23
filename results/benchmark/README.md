# benchmark results

This folder holds the score sheets of the robot benchmark. Each
subfolder is one benchmark set, with the same name as its dataset. The
sets and the results are in [`writeup.md`](../../docs/writeup.md),
section "Final benchmark".

To make the sheets of one set:

1. Record the set with `parcel_vla/benchmark/eval_parcel.sh --set <name>`.
2. Run `python -m parcel_vla.benchmark.bench_sheets <dataset>`.
3. Fill in the outcome columns of `runs/bench/<dataset>/scores.csv`.

Columns of `scores.csv`:

| Column | Meaning |
|---|---|
| `time_s` | Episode length (from the data) |
| `grip_cycles` | Gripper cycles (from the data) |
| `home_s` | Time the arm is still at the rest pose (from the data) |
| `parcels` | Parcels in the heap |
| `cleared` | Parcels off the table |
| `wrong_label` | Parcels that went off the table label up |
| `help` | Times the operator moved a parcel |
| `cases` | Edge cases: `box_on_box`, `far_edge`, `on_table` |
| `note` | Parcels of the heap, and the time of each help |

An episode is a success if all parcels are cleared, with 0 wrong
labels and 0 help.
