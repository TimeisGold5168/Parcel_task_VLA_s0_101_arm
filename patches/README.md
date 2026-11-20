# patches

This folder holds changes to the LeRobot source. The base is LeRobot
commit `64b23178`.

| Patch | Change |
|---|---|
| `lerobot_episodic_skip_empty.patch` | Do not save an empty episode |

The patch changes `src/lerobot/rollout/strategies/episodic.py`. If an
episode of the episodic rollout ends with no frames, the rollout does
not save it and writes a warning.

To apply the patch:

1. Go to the LeRobot root.
2. Set the path:
   `P=outputs/portfolio/patches/lerobot_episodic_skip_empty.patch`.
3. Check the patch: `git apply --check "$P"`.
4. Apply the patch: `git apply "$P"`.
