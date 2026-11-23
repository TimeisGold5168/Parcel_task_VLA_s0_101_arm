# data (phase 02)

This folder holds the tools that record the demos with the leader arm
and 3 cameras, and check them.

| File | Purpose |
|---|---|
| `record_parcel.sh` | Record one heap of a schedule with 3 cameras and the leader arm |
| `record_preview.py` | `lerobot-record` with a large live preview of the 3 cameras |
| `check_episodes.py` | Check the episode lengths, the phase marks and the video frames |
| `make_schedule.py` | Make a balanced, shuffled heap schedule |
| `schedule_A1.csv` | Heaps of dataset A1 (push approach) |
| `schedule_B.csv` | Heaps of dataset B (pick and place, 20 heaps) |
| `schedule_S1.csv` | Heaps of the box-stack skill demos |

## Record dataset B

1. Set your devices (see [`../env.sh`](../env.sh)).
2. Put the arm at the home pose.
3. Build heap 1 as the script shows it.
4. Record the heap:
   `STAGE=b bash parcel_vla/data/record_parcel.sh 1`.
5. In each episode, push Space when the parcel is released label down
   in the flip zone.
6. Record the next heaps: `record_parcel.sh 2`, `3`, and so on.

The script checks the new episodes at the end. To check a dataset
again:
`python -m parcel_vla.data.check_episodes --root <dataset root> --home median`.

Keys: Right arrow ends the episode early, Left arrow records it again,
Esc stops and saves. With the preview, the keys work only when the
preview window has the focus.

## Lessons

- One smooth motion for each parcel type is better than many demos
  with different motions. Dataset B has fewer frames than dataset A1
  (73k against 102k), and it works.
- Use one firm close of the gripper. The model copies hesitant grip
  adjustments.
