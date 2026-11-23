# New-skill checkpoints: offline scores

Roll: start test, old-run start frame (pass >= 15, old model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp pose error 0.5 s before the grasp, deg (lower is better).

## box + far edge (smolvla_b_fb)

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_fb:005000` | 21.4 | 1.00 | 3.1 | 2.9 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 1.4 | 1.4 | yes |
| `smolvla_b_fb:015000` | 14.2 | 1.00 | 1.3 | 1.3 | no |
| `smolvla_b_box` | 19.8 | 1.00 | 3.5 | 1.7 | yes |
| `smolvla_b` | 32.0 | 1.00 | 6.5 | 1.2 | no |

Pick: `smolvla_b_fb:010000`

## far edge only (smolvla_b_fe15)

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_fe15:002500` | 3.1 | 1.00 | 4.1 | 2.4 | no |
| `smolvla_b_fe15:005000` | 23.7 | 1.00 | 2.4 | 1.9 | yes |
| `smolvla_b_fe15:007500` | 26.9 | 1.00 | 1.5 | 1.4 | yes |
| `smolvla_b_fe15:010000` | 31.9 | 1.00 | 1.5 | 1.4 | yes |
| `smolvla_b_fe` | 29.4 | 1.00 | 2.4 | 1.6 | yes |
| `smolvla_b` | 32.0 | 1.00 | 16.3 | 1.2 | no |

Pick: `smolvla_b_fe15:010000`

