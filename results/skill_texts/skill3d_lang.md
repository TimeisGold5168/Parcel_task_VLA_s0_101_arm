# Language test: skill texts (offline, held-out episodes)

Texts: box = "grasp rigid box with wide grip and set it at flip zone"; bag = "pinch soft bag by its edge then drop onto table"; top = "lift upper carton off stacked pile"

Error and shift in deg (5 arm joints, next 1 s). Grip: highest predicted gripper value in the 1 s.

## `smolvla_parcel_b_sk3d_10k:005000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 85% | 3.86 / 2.73 / 5.53 | box 3.13, top 4.72 | 1.90 | 32 / 33 / 37 |
| b_bag | approach | 49 | bag | 92% | 4.12 / 2.64 / 5.73 | box 3.62, top 4.95 | 1.88 | 42 / 42 / 45 |
| b_bag | start | 31 | bag | 74% | 3.44 / 2.87 / 5.22 | box 2.36, top 4.36 | 1.95 | 17 / 19 / 23 |
| b_box | all | 111 | box | 70% | 2.36 / 2.79 / 3.53 | bag 1.29, top 2.40 | 1.73 | 57 / 54 / 59 |
| b_box | approach | 84 | box | 71% | 2.41 / 2.95 / 3.78 | bag 1.44, top 2.71 | 1.81 | 65 / 64 / 67 |
| b_box | start | 27 | box | 67% | 2.20 / 2.30 / 2.76 | bag 0.84, top 1.44 | 1.48 | 31 / 21 / 36 |
| black_box | all | 75 | box | 52% | 5.01 / 5.62 / 6.39 | bag 1.96, top 3.76 | 2.51 | 64 / 63 / 65 |
| black_box | approach | 75 | box | 52% | 5.01 / 5.62 / 6.39 | bag 1.96, top 3.76 | 2.51 | 64 / 63 / 65 |
| stack | all | 80 | top | 30% | 4.42 / 4.35 / 4.50 | box 2.15, bag 2.06 | 2.57 | 65 / 65 / 68 |
| stack | approach | 78 | top | 29% | 4.45 / 4.36 / 4.55 | box 2.16, bag 2.06 | 2.57 | 67 / 67 / 69 |
| stack | start | 2 | top | 50% | 3.27 / 4.02 / 2.72 | box 1.63, bag 2.00 | 2.34 | 3 / 2 / 22 |

## `smolvla_parcel_b_sk3d_10k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 85% | 3.89 / 2.44 / 5.85 | box 3.41, top 5.36 | 1.63 | 31 / 32 / 35 |
| b_bag | approach | 49 | bag | 92% | 4.33 / 2.39 / 6.19 | box 3.85, top 5.65 | 1.57 | 42 / 42 / 44 |
| b_bag | start | 31 | bag | 74% | 3.20 / 2.52 / 5.31 | box 2.70, top 4.90 | 1.74 | 13 / 17 / 20 |
| b_box | all | 111 | box | 75% | 2.09 / 2.56 / 3.51 | bag 1.35, top 2.63 | 1.38 | 55 / 53 / 58 |
| b_box | approach | 84 | box | 74% | 2.17 / 2.72 / 3.78 | bag 1.50, top 2.95 | 1.45 | 65 / 65 / 66 |
| b_box | start | 27 | box | 78% | 1.84 / 2.05 / 2.67 | bag 0.89, top 1.63 | 1.18 | 25 / 15 / 30 |
| black_box | all | 75 | box | 48% | 4.61 / 5.20 / 6.07 | bag 2.11, top 3.96 | 2.08 | 64 / 64 / 65 |
| black_box | approach | 75 | box | 48% | 4.61 / 5.20 / 6.07 | bag 2.11, top 3.96 | 2.08 | 64 / 64 / 65 |
| stack | all | 80 | top | 41% | 4.22 / 4.00 / 4.08 | box 2.25, bag 1.99 | 2.15 | 66 / 66 / 68 |
| stack | approach | 78 | top | 40% | 4.25 / 4.03 / 4.14 | box 2.25, bag 1.99 | 2.15 | 68 / 68 / 69 |
| stack | start | 2 | top | 100% | 3.19 / 2.88 / 1.84 | box 2.22, bag 1.87 | 1.87 | 1 / 2 / 30 |

## `smolvla_parcel_b_all_20k:010000`

| Group | Phase | Frames | Correct text | Wins | Error box / bag / top | Shift (other texts) | Noise | Grip box / bag / top |
|---|---|---|---|---|---|---|---|---|
| b_bag | all | 80 | bag | 91% | 7.28 / 5.39 / 6.72 | box 3.15, top 2.05 | 1.60 | 31 / 32 / 30 |
| b_bag | approach | 49 | bag | 88% | 6.28 / 4.28 / 5.52 | box 3.33, top 2.00 | 1.57 | 42 / 41 / 41 |
| b_bag | start | 31 | bag | 97% | 8.85 / 7.15 / 8.61 | box 2.85, top 2.14 | 1.64 | 13 / 16 / 14 |
| b_box | all | 111 | box | 32% | 5.53 / 5.08 / 6.78 | bag 2.36, top 2.53 | 1.79 | 56 / 57 / 56 |
| b_box | approach | 84 | box | 42% | 4.47 / 4.38 / 5.69 | bag 2.14, top 2.42 | 1.73 | 62 / 63 / 62 |
| b_box | start | 27 | box | 0% | 8.83 / 7.27 / 10.15 | bag 3.02, top 2.86 | 1.98 | 37 / 38 / 37 |
| black_box | all | 75 | box | 28% | 6.00 / 5.16 / 6.85 | bag 3.33, top 3.68 | 3.25 | 62 / 63 / 61 |
| black_box | approach | 75 | box | 28% | 6.00 / 5.16 / 6.85 | bag 3.33, top 3.68 | 3.25 | 62 / 63 / 61 |
| stack | all | 80 | top | 1% | 5.94 / 5.47 / 7.39 | box 3.19, bag 2.62 | 3.13 | 66 / 67 / 65 |
| stack | approach | 78 | top | 1% | 5.77 / 5.31 / 7.16 | box 3.17, bag 2.57 | 3.13 | 66 / 67 / 65 |
| stack | start | 2 | top | 0% | 12.50 / 11.86 / 16.31 | box 3.88, bag 4.53 | 2.98 | 80 / 79 / 79 |
