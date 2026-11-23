# New-skill checkpoints: offline scores

Roll: start test, old-run start frame (pass >= 20, old model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp pose error 0.5 s before the grasp, deg (lower is better).

## black-box corrections (smolvla_bsp_b_all)

Reference `smolvla_bsp_b`: roll >= 9.6, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_bsp_b_all:005000` | 1.6 | 1.00 | 3.6 | 3.7 | no |
| `smolvla_bsp_b_all:010000` | 2.4 | 1.00 | 3.3 | 3.0 | no |
| `smolvla_bsp_b_all:015000` | 3.0 | 1.00 | 3.4 | 2.7 | no |
| `smolvla_bsp_b_all:020000` | 5.2 | 1.00 | 2.7 | 2.5 | no |
| `smolvla_bsp_b_all:025000` | 2.5 | 1.00 | 2.6 | 2.6 | no |
| `smolvla_bsp_b_all:030000` | 3.5 | 1.00 | 2.5 | 2.5 | no |
| `smolvla_bsp_b` | 15.4 | 1.00 | 6.2 | 1.2 | no |

Pick: none passes the rules

## box demos (smolvla_bsp_b_all)

Reference `smolvla_bsp_b`: roll >= 9.6, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_bsp_b_all:005000` | 1.6 | 1.00 | 4.9 | 3.7 | no |
| `smolvla_bsp_b_all:010000` | 2.4 | 1.00 | 3.5 | 3.0 | no |
| `smolvla_bsp_b_all:015000` | 3.0 | 1.00 | 3.2 | 2.7 | no |
| `smolvla_bsp_b_all:020000` | 5.2 | 1.00 | 2.6 | 2.5 | no |
| `smolvla_bsp_b_all:025000` | 2.5 | 1.00 | 2.5 | 2.6 | no |
| `smolvla_bsp_b_all:030000` | 3.5 | 1.00 | 2.4 | 2.5 | no |
| `smolvla_bsp_b` | 15.4 | 1.00 | 6.5 | 1.2 | no |

Pick: none passes the rules

## on-table corrections (smolvla_bsp_b_all)

Reference `smolvla_bsp_b`: roll >= 9.6, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_bsp_b_all:005000` | 1.6 | 1.00 | 4.2 | 3.7 | no |
| `smolvla_bsp_b_all:010000` | 2.4 | 1.00 | 2.8 | 3.0 | no |
| `smolvla_bsp_b_all:015000` | 3.0 | 1.00 | 3.4 | 2.7 | no |
| `smolvla_bsp_b_all:020000` | 5.2 | 1.00 | 3.1 | 2.5 | no |
| `smolvla_bsp_b_all:025000` | 2.5 | 1.00 | 2.8 | 2.6 | no |
| `smolvla_bsp_b_all:030000` | 3.5 | 1.00 | 2.8 | 2.5 | no |
| `smolvla_bsp_b` | 15.4 | 1.00 | 12.9 | 1.2 | no |

Pick: none passes the rules

