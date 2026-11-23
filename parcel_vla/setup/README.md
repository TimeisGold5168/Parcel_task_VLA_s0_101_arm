# setup (phase 01)

This folder holds the tools that prepare the scene before any
training: camera checks and the parcel labels.

| File | Purpose |
|---|---|
| `camera_align.py` | Live front-camera alignment against a recorded episode |
| `focus_guide.py` | Live focus guide for the wrist camera |
| [`labels/`](labels) | ArUco shipping labels, label PDFs and the parcel list |

The snapshots and the focus reference go to `runs/camcheck/`.

```bash
# overlay the live camera on episode 0 of a dataset
python -m parcel_vla.setup.camera_align --device /dev/video6 --ref-episode 0

# focus guide for the wrist camera
python -m parcel_vla.setup.focus_guide
```

Two setup fixes removed failures that more training would not fix:

1. The wrist camera sits one joint back, so the 180° label flip does
   not hit the table.
2. A new base height gives an easier grasp on every parcel.
