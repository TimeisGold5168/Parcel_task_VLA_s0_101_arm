# Spot test: 151-episode models on the real robot (before fixes)

Goal: measure the three models trained on all 151 episodes on the
same spots, before fine-tuning or DAgger. Results go in
`runs/results_spots.csv` (filled automatically, then checked).

Models (all trained on third-act-attempt, 151 episodes):
- act151: ACT, act_all_ft50k checkpoint 050000 (with ensembling)
- diffusion: diffusion_all_100k checkpoint 100000
- smolvla: smolvla_all_100k checkpoint 100000

## 0. Setup (once, before the first spot)
1. Close any old rerun viewer and rollout.
2. Align the front camera to episode 0 (pose A):
   ```bash
   cd ~/lerobot/lerobot
   python -m parcel_vla.setup.camera_align --device /dev/video6 \
     --ref-episode 0
   ```
   Adjust until the text is green, press `s` (snapshot), then `q`.
   Lock the camera. Only move it at the planned re-alignment.
3. Bin, tape square and X marks as during recording. Same lighting.
4. Tape three small unseen-spot marks (UPL, UPM, UPR) at places
   never used in recording, e.g. left / middle / right between the
   X marks. Take one photo of the layout: `spots_layout.jpg`.

Camera poses in the training data (third-act-attempt):
- pose A: episodes 0-70 (all square episodes)
- pose B: episodes 71-95 (about 6 deg off; 15 X-mark episodes)
- pose C: episodes 96-150 (about 3 deg off; 55 X-mark episodes)

## 1. Spots (3 trials per model per spot)
| Spot | Where the cube goes                        | Pose |
|------|--------------------------------------------|------|
| SQ   | centre of the blue square, as recorded     | A    |
| SQR  | random spot inside the square, new each    | A    |
|      | trial                                      |      |
| X1-5 | on the X mark                              | C    |
| UPL  | unseen spot, left                          | C    |
| UPM  | unseen spot, middle                        | C    |
| UPR  | unseen spot, right                         | C    |

Every spot is run by act151, diffusion and smolvla, plus a
distractor version for smolvla only: `<spot>D` (e.g. SQD, X1D, UPLD)
= same green cube position with a blue or red cube placed next to it.

Total: 10 spots x (3 models + smolvla D) x 3 trials = 120 trials.
Keep the cube orientation the same as during recording.

## 2. Running a spot
One command per spot runs every model in turn
(act151 -> diffusion -> smolvla -> smolvla with distractor):
```bash
cd ~/lerobot/lerobot
bash parcel_vla/cube_warmup/run_spot.sh SQ
```
Before each model it waits for Enter (type `s` to skip a model).
After each model it auto-scores the trials into `runs/results_spots.csv`
and saves a review sheet in `runs/spot_review/`.

Important: when a model finishes, the arm's torque turns off and it
goes limp. Hold or fold it to the home pose before pressing Enter;
the pose at startup becomes the home the arm returns to between
trials.

Phase 1 - camera at pose A (`--ref-episode 0`):
```bash
bash parcel_vla/cube_warmup/run_spot.sh SQ
bash parcel_vla/cube_warmup/run_spot.sh SQR
```
Phase 2 - re-align camera to pose C (`--ref-episode 130`), lock it:
```bash
for s in X1 X2 X3 X4 X5 UPL UPM UPR; do
  bash parcel_vla/cube_warmup/run_spot.sh $s
done
```
Save an alignment snapshot (`s`) at the start of each phase.
Recordings: `~/.cache/huggingface/lerobot/Low101/rollout_<model>_<spot>_*`

Controls during a session: Right arrow = end episode early (success
or clear failure), Left arrow = redo episode, Esc = stop.
Wait until "Recording episode" before pressing a key.

Time: about 1 min per trial incl. reset plus model loading
-> about 2.5 h in total.

## 3. Scoring (automatic, `parcel_vla/cube_warmup/review_rollouts.py`)
- `stage_reached`: none / reach / grasp / lift / place
- `task_success`: 1 if place, else 0. The gripper opens after the
  last hold and the cube is in the bin.
- `pick_success`: 1 if lift or place, else 0. The gripper holds the
  cube and the cube is off the table.
- `miss_direction`: where the first close landed relative to the
  cube: `toward_square`, `away_square`, `left`, `right`, `short`,
  `long`, or `none` if on the cube
- `miss_cm`: distance of that landing point from the cube
- `notes`: attempts, hold time, `check` if the tool was unsure

Manual edits are kept: the tool only fills rows that are empty.

## 4. What the results will tell us
| Pattern                                     | Likely cause        |
|---------------------------------------------|---------------------|
| act151/smolvla miss toward square at X,     | data imbalance /    |
| diffusion hits                              | ACT averaging       |
| all models miss in the same direction       | camera pose / setup |
| SQ ok, SQR worse                            | needs more varied   |
|                                             | square data         |
| X ok, UP spots fail                         | memorised spots, no |
|                                             | real generalisation |
| smolvla <spot>D fails where <spot> works    | colour/distractor   |
|                                             | confusion           |

Next step after this: fine-tune ACT (balanced + augmentation), then
DAgger on whatever still fails, then rerun this exact test.
