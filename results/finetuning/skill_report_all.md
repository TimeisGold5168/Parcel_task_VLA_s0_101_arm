# New-skill checkpoints: offline scores

Roll: start test, old-run start frame (pass >= 20, old model 32). Grip: gripper opens at 1.6 s. Fit: mean grasp pose error 0.5 s before the grasp, deg (lower is better).

## black-box corrections (smolvla_b_all)

Reference `smolvla_b`: roll >= 20.0, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_all:005000` | 22.5 | 1.00 | 3.7 | 2.1 | yes |
| `smolvla_b_all:010000` | 22.9 | 1.00 | 2.8 | 1.3 | yes |
| `smolvla_b_all:015000` | 10.5 | 1.00 | 2.6 | 1.4 | no |
| `smolvla_b_all:020000` | 8.2 | 1.00 | 2.4 | 1.3 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 6.8 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 6.3 | 1.2 | no |

Pick: `smolvla_b_all:010000`

## box demos (smolvla_b_all)

Reference `smolvla_b`: roll >= 20.0, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_all:005000` | 22.5 | 1.00 | 1.6 | 2.1 | yes |
| `smolvla_b_all:010000` | 22.9 | 1.00 | 2.0 | 1.3 | yes |
| `smolvla_b_all:015000` | 10.5 | 1.00 | 1.6 | 1.4 | no |
| `smolvla_b_all:020000` | 8.2 | 1.00 | 1.5 | 1.3 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 1.4 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 6.5 | 1.2 | no |

Pick: `smolvla_b_fb:010000`

## far-edge picks (smolvla_b_all)

Reference `smolvla_b`: roll >= 20.0, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_all:005000` | 22.5 | 1.00 | 1.7 | 2.1 | yes |
| `smolvla_b_all:010000` | 22.9 | 1.00 | 1.5 | 1.3 | yes |
| `smolvla_b_all:015000` | 10.5 | 1.00 | 1.3 | 1.4 | no |
| `smolvla_b_all:020000` | 8.2 | 1.00 | 1.1 | 1.3 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 1.9 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 16.3 | 1.2 | no |

Pick: `smolvla_b_all:010000`

## stacked-parcel corrections (smolvla_b_all)

Reference `smolvla_b`: roll >= 20.0, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_all:005000` | 22.5 | 1.00 | 2.0 | 2.1 | yes |
| `smolvla_b_all:010000` | 22.9 | 1.00 | 1.8 | 1.3 | yes |
| `smolvla_b_all:015000` | 10.5 | 1.00 | 1.8 | 1.4 | no |
| `smolvla_b_all:020000` | 8.2 | 1.00 | 1.6 | 1.3 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 3.5 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 3.8 | 1.2 | no |

Pick: `smolvla_b_all:010000`

## on-table corrections (smolvla_b_all)

Reference `smolvla_b`: roll >= 20.0, B fit <= 2.2.

| Checkpoint | Roll | Grip | New-skill fit | B fit | Passes rules 1-2 |
|---|---|---|---|---|---|
| `smolvla_b_all:005000` | 22.5 | 1.00 | 4.9 | 2.1 | yes |
| `smolvla_b_all:010000` | 22.9 | 1.00 | 3.8 | 1.3 | yes |
| `smolvla_b_all:015000` | 10.5 | 1.00 | 3.1 | 1.4 | no |
| `smolvla_b_all:020000` | 8.2 | 1.00 | 3.0 | 1.3 | no |
| `smolvla_b_fb:010000` | 21.1 | 1.00 | 12.4 | 1.4 | yes |
| `smolvla_b` | 32.0 | 1.00 | 12.9 | 1.2 | no |

Pick: `smolvla_b_all:010000`

