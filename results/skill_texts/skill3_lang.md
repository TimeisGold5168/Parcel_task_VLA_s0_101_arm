# Language test: skill texts (offline, held-out episodes)

Texts: box = "pick the box and place it in the flip zone"; bag = "pick the bag and place it in the flip zone"; top = "pick the top box off the stack"

Error and shift in deg (5 arm joints, next 1 s). Grip: highest predicted gripper value in the 1 s.

## `smolvla_parcel_b_sk3_10k:005000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 64% | 2.71 / 2.66 / 4.30 | box 0.37, top 2.87 | 1.92 | 34 / 33 / 35 |
| b_bag | approach | 49 | bag | 63% | 2.66 / 2.62 / 4.48 | box 0.32, top 3.29 | 1.92 | 43 / 43 / 44 |
| b_bag | start | 31 | bag | 65% | 2.79 / 2.73 / 4.01 | box 0.45, top 2.22 | 1.92 | 19 / 18 / 22 |
| b_box | all | 111 | box | 55% | 2.32 / 2.36 / 3.25 | bag 0.27, top 1.96 | 1.68 | 57 / 56 / 60 |
| b_box | approach | 84 | box | 45% | 2.36 / 2.38 / 3.43 | bag 0.27, top 2.21 | 1.73 | 65 / 65 / 67 |
| b_box | start | 27 | box | 85% | 2.18 / 2.29 / 2.70 | bag 0.27, top 1.17 | 1.51 | 31 / 29 / 38 |
| black_box | all | 75 | box | 35% | 4.98 / 4.99 / 6.62 | bag 0.29, top 4.01 | 2.41 | 64 / 64 / 65 |
| black_box | approach | 75 | box | 35% | 4.98 / 4.99 / 6.62 | bag 0.29, top 4.01 | 2.41 | 64 / 64 / 65 |
| stack | all | 80 | top | 38% | 4.31 / 4.30 / 4.53 | box 1.87, bag 1.82 | 2.54 | 66 / 66 / 68 |
| stack | approach | 78 | top | 37% | 4.31 / 4.30 / 4.55 | box 1.87, bag 1.82 | 2.53 | 68 / 68 / 69 |
| stack | start | 2 | top | 50% | 4.37 / 4.24 / 3.50 | box 1.81, bag 1.64 | 2.59 | 5 / 4 / 20 |

## `smolvla_parcel_b_sk3_10k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 62% | 2.44 / 2.40 / 4.18 | box 0.40, top 2.97 | 1.62 | 33 / 32 / 34 |
| b_bag | approach | 49 | bag | 59% | 2.46 / 2.43 / 4.43 | box 0.33, top 3.33 | 1.59 | 43 / 42 / 43 |
| b_bag | start | 31 | bag | 68% | 2.41 / 2.35 / 3.79 | box 0.52, top 2.41 | 1.68 | 17 / 17 / 19 |
| b_box | all | 111 | box | 57% | 2.09 / 2.11 / 3.17 | bag 0.28, top 2.16 | 1.35 | 55 / 54 / 59 |
| b_box | approach | 84 | box | 54% | 2.16 / 2.18 / 3.39 | bag 0.29, top 2.46 | 1.40 | 65 / 65 / 66 |
| b_box | start | 27 | box | 67% | 1.84 / 1.90 / 2.47 | bag 0.26, top 1.23 | 1.18 | 23 / 22 / 35 |
| black_box | all | 75 | box | 24% | 4.61 / 4.61 / 6.49 | bag 0.32, top 4.52 | 1.99 | 64 / 64 / 65 |
| black_box | approach | 75 | box | 24% | 4.61 / 4.61 / 6.49 | bag 0.32, top 4.52 | 1.99 | 64 / 64 / 65 |
| stack | all | 80 | top | 54% | 4.09 / 4.04 / 4.06 | box 1.90, bag 1.84 | 2.11 | 67 / 66 / 68 |
| stack | approach | 78 | top | 53% | 4.11 / 4.06 / 4.11 | box 1.89, bag 1.84 | 2.11 | 68 / 68 / 69 |
| stack | start | 2 | top | 100% | 3.41 / 3.21 / 2.15 | box 2.22, bag 1.92 | 1.85 | 4 / 3 / 33 |

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
