# New-skill checkpoints: offline scores

Roll: start test, old-run start frame (pass >= 15, old model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp pose error 0.5 s before the grasp, deg (lower is better).

## stacked-parcel corrections (smolvla_b_fbs)

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_fbs:002500` | 17.6 | 1.00 | 2.5 | 3.1 | no |
| `smolvla_b_fbs:005000` | 15.2 | 1.00 | 2.1 | 2.2 | no |
| `smolvla_b_fbs:007500` | 16.2 | 1.00 | 1.8 | 1.9 | yes |
| `smolvla_b_fbs:010000` | 18.3 | 1.00 | 1.8 | 1.7 | yes |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 3.5 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 4.8 | 1.2 | no |

Pick: `smolvla_b_fbs:007500`

## box demos, kept? (smolvla_b_fbs)

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_fbs:002500` | 17.6 | 1.00 | 2.3 | 3.1 | no |
| `smolvla_b_fbs:005000` | 15.2 | 1.00 | 1.9 | 2.2 | no |
| `smolvla_b_fbs:007500` | 16.2 | 1.00 | 1.6 | 1.9 | yes |
| `smolvla_b_fbs:010000` | 18.3 | 1.00 | 1.3 | 1.7 | yes |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 1.4 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 6.5 | 1.2 | no |

Pick: `smolvla_b_fbs:010000`

