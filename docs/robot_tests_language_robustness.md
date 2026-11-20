# Robot tests: language and robustness

Run from `~/lerobot/lerobot` in the `lerobot` conda env. All runs use
`outputs/portfolio/run_robot_test.sh`. Results go into
`results_language_robustness.csv` (one row per episode).

## Before starting

1. Close anything using the GPU (no training running).
2. Check the camera devices: wrist = `/dev/video4`, front = `/dev/video6`.
   If they changed, prefix commands with
   `WRIST_DEV=/dev/videoX FRONT_DEV=/dev/videoY`.
3. Same layout as recording: bin in its taped square, X marks on the
   table, arm in the home pose before every episode.
4. Test spots used everywhere below (3 per run, in this order):
   blue square, X mark nearest the robot, X mark far right.

## Part A: language test (SmolVLA only, ~20 min)

| Run | Table | Instruction | What to note |
|---|---|---|---|
| A1 | green cube | training instruction (default) | baseline |
| A2 | green cube | "pick the red cube and place into the bin" | still picks green? |
| A3 | green cube + red object | "pick the red cube and place into the bin" | which object it goes for |
| A4 | green cube | "wave at the camera" | does it still do the task? |

```bash
P=outputs/portfolio/run_robot_test.sh
$P smolvla lang_normal 3
$P smolvla lang_red 3 "pick the red cube and place into the bin"
$P smolvla lang_red_both 3 "pick the red cube and place into the bin"
$P smolvla lang_wave 3 "wave at the camera"
```

Expected from the offline test: "red cube" changes predictions about as
little as random noise, so it will most likely still pick the green
cube. Unrelated text changes predictions much more.

## Part B: robustness (all 3 models, ~1.5 h)

Conditions (3 spots each):

| Condition | Change |
|---|---|
| normal | nothing changed |
| distractors | 2-3 objects never seen before on the table, not touching the cube or the path to the bin |
| lamp | one room light off, or a desk lamp shining from the side |
| new_cube | a slightly different cube (other size or shade), if available |

```bash
P=outputs/portfolio/run_robot_test.sh
for M in act diffusion smolvla; do
  $P $M normal 3
done
# set up distractors, then:
for M in act diffusion smolvla; do $P $M distractors 3; done
# change lighting, then:
for M in act diffusion smolvla; do $P $M lamp 3; done
# swap the cube, then:
for M in act diffusion smolvla; do $P $M new_cube 3; done
```

Run each condition for all three models before changing the scene, so
every model sees exactly the same setup.

## Scoring

For every episode record the last stage reached:
none / reach / grasp / lift / place (place = success). For the language
test also note which object the arm went for.
