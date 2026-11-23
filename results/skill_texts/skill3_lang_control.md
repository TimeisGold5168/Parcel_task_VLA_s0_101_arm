# Language test: skill texts (offline, held-out episodes)

Texts: box = "pick the box and place it in the flip zone"; bag = "pick the bag and place it in the flip zone"; top = "pick the top box off the stack"

Error and shift in deg (5 arm joints, next 1 s). Grip: highest predicted gripper value in the 1 s.

## `smolvla_parcel_b_all_20k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 8% | 4.56 / 4.53 / 4.04 | box 0.27, top 1.35 | 1.59 | 30 / 30 / 30 |
| b_bag | approach | 49 | bag | 6% | 3.46 / 3.43 / 2.96 | box 0.25, top 1.19 | 1.48 | 41 / 40 / 40 |
| b_bag | start | 31 | bag | 10% | 6.31 / 6.27 / 5.74 | box 0.30, top 1.61 | 1.76 | 13 / 13 / 14 |
| b_box | all | 111 | box | 24% | 4.61 / 4.55 / 4.02 | bag 0.32, top 1.34 | 1.77 | 54 / 54 / 55 |
| b_box | approach | 84 | box | 31% | 3.80 / 3.78 / 3.53 | bag 0.30, top 1.10 | 1.67 | 62 / 62 / 63 |
| b_box | start | 27 | box | 4% | 7.11 / 6.95 / 5.55 | bag 0.36, top 2.11 | 2.08 | 30 / 31 / 33 |
| black_box | all | 75 | box | 25% | 5.15 / 5.07 / 5.23 | bag 0.31, top 1.57 | 2.85 | 62 / 62 / 62 |
| black_box | approach | 75 | box | 25% | 5.15 / 5.07 / 5.23 | bag 0.31, top 1.57 | 2.85 | 62 / 62 / 62 |
| stack | all | 80 | top | 30% | 4.87 / 4.83 / 5.02 | box 1.14, bag 1.12 | 2.55 | 68 / 67 / 68 |
| stack | approach | 78 | top | 28% | 4.76 / 4.73 / 4.96 | box 1.11, bag 1.10 | 2.57 | 67 / 67 / 67 |
| stack | start | 2 | top | 100% | 9.24 / 8.93 / 7.37 | box 2.24, bag 1.97 | 1.87 | 77 / 77 / 78 |
