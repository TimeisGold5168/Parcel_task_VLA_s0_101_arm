# Language test: skill texts (offline, held-out episodes)

Texts: box = "grasp rigid box with wide grip and set it at flip zone"; bag = "pinch soft bag by its edge then drop onto table"; top = "lift upper carton off stacked pile"; far = "reach far rim, pull parcel closer"; push = "sweep lone item beyond wooden desk border"

Error and shift in deg (5 arm joints, next 1 s). Grip: highest predicted gripper value in the 1 s.

## `smolvla_parcel_b_sk5d_10k:005000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top / far / push | Shift (other texts) | Noise | Grip box / bag / top / far / push |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 78% | 4.15 / 2.82 / 6.27 / 4.57 / 5.13 | box 3.45, top 5.49, far 3.65, push 4.17 | 2.06 | 33 / 34 / 38 / 33 / 29 |
| b_bag | approach | 49 | bag | 84% | 4.44 / 2.63 / 6.77 / 4.80 / 5.00 | box 3.85, top 6.14, far 4.01, push 4.41 | 2.02 | 43 / 43 / 47 / 44 / 43 |
| b_bag | start | 31 | bag | 68% | 3.69 / 3.13 / 5.48 / 4.20 / 5.33 | box 2.80, top 4.47, far 3.09, push 3.79 | 2.12 | 16 / 18 / 23 / 16 / 7 |
| b_box | all | 111 | box | 68% | 2.61 / 3.03 / 4.13 / 3.91 / 3.45 | bag 1.25, top 2.52, far 2.12, push 1.86 | 1.77 | 58 / 55 / 63 / 60 / 56 |
| b_box | approach | 84 | box | 75% | 2.71 / 3.25 / 4.40 / 4.22 / 3.60 | bag 1.38, top 2.77, far 2.25, push 1.79 | 1.87 | 67 / 66 / 69 / 67 / 66 |
| b_box | start | 27 | box | 48% | 2.29 / 2.35 / 3.26 / 2.96 / 2.98 | bag 0.83, top 1.74, far 1.72, push 2.07 | 1.44 | 32 / 21 / 42 / 38 / 24 |
| black_box | all | 75 | box | 40% | 5.00 / 5.43 / 6.14 / 6.05 / 5.60 | bag 1.81, top 3.16, far 2.42, push 1.86 | 2.58 | 65 / 65 / 67 / 65 / 65 |
| black_box | approach | 75 | box | 40% | 5.00 / 5.43 / 6.14 / 6.05 / 5.60 | bag 1.81, top 3.16, far 2.42, push 1.86 | 2.58 | 65 / 65 / 67 / 65 / 65 |
| far_edge | all | 86 | far | 41% | 5.90 / 5.70 / 6.39 / 5.52 / 5.52 | box 2.60, bag 3.31, top 2.80, push 2.42 | 3.39 | 60 / 60 / 60 / 61 / 59 |
| far_edge | approach | 86 | far | 41% | 5.90 / 5.70 / 6.39 / 5.52 / 5.52 | box 2.60, bag 3.31, top 2.80, push 2.42 | 3.39 | 60 / 60 / 60 / 61 / 59 |
| on_table | all | 13 | push | 0% | 10.82 / 12.24 / 11.24 / 12.83 / 12.54 | box 4.36, bag 5.68, top 3.76, far 3.61 | 4.59 | 48 / 50 / 56 / 50 / 46 |
| on_table | approach | 13 | push | 0% | 10.82 / 12.24 / 11.24 / 12.83 / 12.54 | box 4.36, bag 5.68, top 3.76, far 3.61 | 4.59 | 48 / 50 / 56 / 50 / 46 |
| stack | all | 80 | top | 15% | 4.53 / 4.19 / 4.55 / 4.78 / 4.60 | box 2.35, bag 1.93, far 1.96, push 1.65 | 2.60 | 67 / 67 / 71 / 67 / 66 |
| stack | approach | 78 | top | 14% | 4.53 / 4.20 / 4.58 / 4.74 / 4.60 | box 2.36, bag 1.94, far 1.91, push 1.64 | 2.60 | 68 / 68 / 71 / 68 / 68 |
| stack | start | 2 | top | 50% | 4.46 / 3.80 / 3.54 / 6.35 / 4.80 | box 1.87, bag 1.65, far 3.64, push 1.82 | 2.46 | 11 / 2 / 62 / 28 / 4 |

## `smolvla_parcel_b_sk5d_10k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top / far / push | Shift (other texts) | Noise | Grip box / bag / top / far / push |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 89% | 4.16 / 2.38 / 6.54 / 4.45 / 4.88 | box 3.49, top 5.97, far 3.65, push 4.09 | 1.61 | 30 / 32 / 34 / 31 / 28 |
| b_bag | approach | 49 | bag | 96% | 4.66 / 2.27 / 7.29 / 4.82 / 5.10 | box 3.98, top 6.78, far 4.08, push 4.53 | 1.56 | 42 / 42 / 44 / 42 / 42 |
| b_bag | start | 31 | bag | 77% | 3.37 / 2.54 / 5.35 / 3.86 / 4.52 | box 2.71, top 4.69, far 2.98, push 3.39 | 1.69 | 12 / 17 / 17 / 13 / 5 |
| b_box | all | 111 | box | 74% | 2.19 / 2.69 / 3.70 / 3.31 / 3.21 | bag 1.27, top 2.64, far 1.95, push 1.93 | 1.43 | 55 / 52 / 58 / 56 / 52 |
| b_box | approach | 84 | box | 79% | 2.29 / 2.88 / 3.96 / 3.53 / 3.37 | bag 1.40, top 2.89, far 2.06, push 1.95 | 1.52 | 65 / 65 / 67 / 64 / 64 |
| b_box | start | 27 | box | 59% | 1.87 / 2.09 / 2.89 / 2.63 / 2.70 | bag 0.85, top 1.86, far 1.61, push 1.88 | 1.17 | 23 / 14 / 32 / 29 / 15 |
| black_box | all | 75 | box | 43% | 4.54 / 5.11 / 5.74 / 5.39 / 5.23 | bag 1.98, top 3.51, far 2.30, push 2.05 | 2.13 | 64 / 64 / 65 / 64 / 64 |
| black_box | approach | 75 | box | 43% | 4.54 / 5.11 / 5.74 / 5.39 / 5.23 | bag 1.98, top 3.51, far 2.30, push 2.05 | 2.13 | 64 / 64 / 65 / 64 / 64 |
| far_edge | all | 86 | far | 34% | 5.31 / 5.48 / 5.36 / 4.89 / 5.21 | box 2.58, bag 2.81, top 3.11, push 2.01 | 2.69 | 58 / 58 / 58 / 60 / 58 |
| far_edge | approach | 86 | far | 34% | 5.31 / 5.48 / 5.36 / 4.89 / 5.21 | box 2.58, bag 2.81, top 3.11, push 2.01 | 2.69 | 58 / 58 / 58 / 60 / 58 |
| on_table | all | 13 | push | 0% | 9.80 / 11.28 / 9.82 / 11.26 / 10.69 | box 3.68, bag 5.85, top 3.55, far 3.52 | 4.08 | 45 / 47 / 50 / 46 / 43 |
| on_table | approach | 13 | push | 0% | 9.80 / 11.28 / 9.82 / 11.26 / 10.69 | box 3.68, bag 5.85, top 3.55, far 3.52 | 4.08 | 45 / 47 / 50 / 46 / 43 |
| stack | all | 80 | top | 19% | 4.22 / 3.98 / 4.13 / 4.25 / 4.27 | box 2.22, bag 1.76, far 2.03, push 1.74 | 2.17 | 66 / 66 / 69 / 66 / 66 |
| stack | approach | 78 | top | 17% | 4.24 / 4.01 / 4.18 / 4.24 / 4.29 | box 2.22, bag 1.76, far 1.99, push 1.74 | 2.18 | 68 / 68 / 69 / 67 / 67 |
| stack | start | 2 | top | 100% | 3.32 / 3.03 / 2.11 / 4.66 / 3.27 | box 2.21, bag 1.80, far 3.54, push 1.74 | 1.72 | 2 / 2 / 62 / 10 / 2 |

## `smolvla_parcel_b_all_20k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top / far / push | Shift (other texts) | Noise | Grip box / bag / top / far / push |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 78% | 7.28 / 5.39 / 6.72 / 6.92 / 7.68 | box 3.15, top 2.05, far 2.44, push 3.02 | 1.60 | 31 / 32 / 30 / 25 / 33 |
| b_bag | approach | 49 | bag | 65% | 6.28 / 4.28 / 5.52 / 5.28 / 6.36 | box 3.33, top 2.00, far 2.23, push 2.98 | 1.57 | 42 / 41 / 41 / 34 / 42 |
| b_bag | start | 31 | bag | 97% | 8.85 / 7.15 / 8.61 / 9.52 / 9.76 | box 2.85, top 2.14, far 2.77, push 3.07 | 1.64 | 13 / 16 / 14 / 11 / 19 |
| b_box | all | 111 | box | 32% | 5.53 / 5.08 / 6.78 / 6.72 / 7.21 | bag 2.36, top 2.53, far 2.60, push 2.48 | 1.79 | 56 / 57 / 56 / 54 / 57 |
| b_box | approach | 84 | box | 42% | 4.47 / 4.38 / 5.69 / 5.60 / 5.92 | bag 2.14, top 2.42, far 2.39, push 2.07 | 1.73 | 62 / 63 / 62 / 60 / 62 |
| b_box | start | 27 | box | 0% | 8.83 / 7.27 / 10.15 / 10.20 / 11.23 | bag 3.02, top 2.86, far 3.26, push 3.79 | 1.98 | 37 / 38 / 37 / 34 / 42 |
| black_box | all | 75 | box | 23% | 6.00 / 5.16 / 6.85 / 6.24 / 6.48 | bag 3.33, top 3.68, far 2.77, push 2.80 | 3.25 | 62 / 63 / 61 / 60 / 62 |
| black_box | approach | 75 | box | 23% | 6.00 / 5.16 / 6.85 / 6.24 / 6.48 | bag 3.33, top 3.68, far 2.77, push 2.80 | 3.25 | 62 / 63 / 61 / 60 / 62 |
| far_edge | all | 86 | far | 0% | 7.09 / 9.29 / 12.21 / 11.74 / 10.06 | box 5.64, bag 3.42, top 2.61, push 2.42 | 3.49 | 56 / 56 / 54 / 54 / 55 |
| far_edge | approach | 86 | far | 0% | 7.09 / 9.29 / 12.21 / 11.74 / 10.06 | box 5.64, bag 3.42, top 2.61, push 2.42 | 3.49 | 56 / 56 / 54 / 54 / 55 |
| on_table | all | 13 | push | 0% | 14.16 / 12.31 / 14.98 / 14.95 / 15.89 | box 4.51, bag 4.92, top 3.83, far 3.58 | 6.66 | 34 / 34 / 32 / 31 / 36 |
| on_table | approach | 13 | push | 0% | 14.16 / 12.31 / 14.98 / 14.95 / 15.89 | box 4.51, bag 4.92, top 3.83, far 3.58 | 6.66 | 34 / 34 / 32 / 31 / 36 |
| stack | all | 80 | top | 1% | 5.94 / 5.47 / 7.39 / 7.05 / 7.36 | box 3.19, bag 2.62, far 1.87, push 2.24 | 3.13 | 66 / 67 / 65 / 64 / 65 |
| stack | approach | 78 | top | 1% | 5.77 / 5.31 / 7.16 / 6.80 / 7.03 | box 3.17, bag 2.57, far 1.84, push 2.19 | 3.13 | 66 / 67 / 65 / 64 / 65 |
| stack | start | 2 | top | 0% | 12.50 / 11.86 / 16.31 / 16.65 / 20.34 | box 3.88, bag 4.53, far 2.83, push 4.11 | 2.98 | 80 / 79 / 79 / 75 / 75 |
