# New-skill checkpoints: offline scores

Roll: start test, old-run start frame (pass >= 15, old model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp pose error 0.5 s before the grasp, deg (lower is better).

## far-edge picks (smolvla_bsp_b_fb)

Reference `smolvla_bsp_b`: roll >= 7.2, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_bsp_b_fb:005000` | 5.6 | 1.00 | 6.0 | 3.7 | no |
| `smolvla_bsp_b_fb:010000` | 18.0 | 1.00 | 2.1 | 2.5 | no |
| `smolvla_bsp_b_fb:015000` | 3.5 | 1.00 | nan | 2.5 | no |
| `smolvla_bsp_b_fb:020000` | 24.5 | 1.00 | nan | 2.0 | yes |
| `smolvla_bsp_b_fb:025000` | 25.7 | 1.00 | nan | 1.8 | yes |
| `smolvla_bsp_b_fb:030000` | 8.2 | 1.00 | nan | 1.8 | yes |
| `smolvla_bsp_b` | 15.4 | 1.00 | 15.7 | 1.2 | no |

Pick: `smolvla_bsp_b_fb:020000`

## box demos (smolvla_bsp_b_fb)

Reference `smolvla_bsp_b`: roll >= 7.2, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_bsp_b_fb:005000` | 5.6 | 1.00 | 4.3 | 3.7 | no |
| `smolvla_bsp_b_fb:010000` | 18.0 | 1.00 | 2.5 | 2.5 | no |
| `smolvla_bsp_b_fb:015000` | 3.5 | 1.00 | 2.0 | 2.5 | no |
| `smolvla_bsp_b_fb:020000` | 24.5 | 1.00 | 1.5 | 2.0 | yes |
| `smolvla_bsp_b_fb:025000` | 25.7 | 1.00 | 1.2 | 1.8 | yes |
| `smolvla_bsp_b_fb:030000` | 8.2 | 1.00 | 1.0 | 1.8 | yes |
| `smolvla_bsp_b` | 15.4 | 1.00 | 6.5 | 1.2 | no |

Pick: `smolvla_bsp_b_fb:030000`

