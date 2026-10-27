# Fixed evaluation protocol (SO-101, pick green cube -> bin)

Use the same protocol for every model (ACT, ACT + corrections,
Diffusion, SmolVLA, RL, B-spline) so results are comparable.

## Setup (keep identical across all models)
- Bin taped in the same place as during recording.
- Front camera untouched since recording; check the "front" stream
  in rerun really shows the stand view.
- Arm starts from the recording home pose (fold it by hand / with
  the leader before each trial).
- Same lighting; no extra objects unless running the distractor test.

## Test spots
Tape 10 small marks on the table and label them.
- S1-S5 ("seen"): spots you used while recording the 51 demos.
- N1-N5 ("new"): spots never used in the demos, e.g. between and
  slightly outside the seen spots.

Draw/photograph the layout once and save it as `spots_layout.jpg`.

## Trials
- 2 trials per spot = 20 trials per model.
- Order: S1 N1 S2 N2 S3 N3 S4 N4 S5 N5, then repeat the same order.
- Trial number = recorded episode index + 1, so videos map to rows.
- Max 30 s per trial. Press Right arrow to end early once the cube
  is in the bin or the attempt has clearly failed.

## Scoring (fill results.csv)
Record how far each trial got (stages, in order):
1. reach   - gripper gets to the cube
2. grasp   - cube closed in gripper
3. lift    - cube lifted off the table
4. place   - cube released inside the bin (= success)

`stage_reached` = last stage completed (none/reach/grasp/lift/place).
`success` = 1 only if stage_reached is place.

## Report
- Success rate on seen spots, new spots, and overall.
- Stage-by-stage counts (shows where each model fails).
- One short note per failure (e.g. "reached left of cube").
