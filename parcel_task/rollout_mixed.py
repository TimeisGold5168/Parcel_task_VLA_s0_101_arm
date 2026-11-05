"""lerobot-rollout with the mixed-camera-size Diffusion patch.

Same arguments as lerobot-rollout. Diffusion stacks the camera images
before the resize, so the 848x480 top and 640x480 wrist/side views need
patch_diffusion() (see train_mixed.py). Other policies are unchanged.

fast_prepare() replaces prepare_observation_for_inference: the stock
one converts the 3 camera images to float on the CPU every tick, which
competes with the video encoder threads (loop 10-15 Hz, not 30 Hz).
Here the uint8 images go to the GPU first and are converted there.

PREVIEW=1 (default) shows the cameras in one mpv window, as in
record_preview.py, with the control loop rate. Keys go to LeRobot's
global listener (Right, Left, Esc), with any window in focus.
Env: PREVIEW, PREVIEW_W (default 1600), PREVIEW_FPS (default 15),
PREVIEW_LABEL (text on the window), SMOOTH (SmolVLA chunk moving
average over this many actions; 0 = off).
"""

import collections
import os
import time

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from train_mixed import patch_diffusion


def fast_prepare(observation, device, task=None, robot_type=None):
    for name in observation:
        x = torch.from_numpy(observation[name]).to(device, non_blocking=True)
        if "image" in name:
            if x.dtype == torch.uint8:
                x = x.float().div_(255)
            x = x.permute(2, 0, 1).contiguous()
        observation[name] = x.unsqueeze(0)
    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return observation


def patch_prepare():
    from lerobot.rollout.inference import rtc, sync
    for mod in (sync, rtc):
        if hasattr(mod, "prepare_observation_for_inference"):
            mod.prepare_observation_for_inference = fast_prepare


def patch_smooth(k):
    """Moving average over k actions of each SmolVLA chunk (k odd).

    SmolVLA 20k chunks are ~20x rougher than the demos (2nd difference);
    k=7 cuts that ~8x with no loss of offline accuracy.
    """
    import torch.nn.functional as F
    from lerobot.policies.smolvla import modeling_smolvla as ms
    orig = ms.SmolVLAPolicy._get_action_chunk

    def smoothed(self, *a, **kw):
        x = orig(self, *a, **kw)  # (B, T, D)
        y = F.pad(x.transpose(1, 2).float(), (k // 2, k // 2),
                  mode="replicate")
        y = F.avg_pool1d(y, k, stride=1).transpose(1, 2)
        return y.to(x.dtype)

    ms.SmolVLAPolicy._get_action_chunk = smoothed


def start_preview(robot, label=None):
    """Camera preview of a connected robot, with its send_action rate.

    Does not take the keyboard focus (the terminal keeps it).
    """
    # record_preview patches LeRobotDataset for phase marks on import;
    # undo that, only its Preview window is used here
    originals = (LeRobotDataset.add_frame, LeRobotDataset.save_episode,
                 LeRobotDataset.clear_episode_buffer)
    import record_preview as rp
    (LeRobotDataset.add_frame, LeRobotDataset.save_episode,
     LeRobotDataset.clear_episode_buffer) = originals

    ticks = collections.deque(maxlen=60)
    label = label or os.environ.get("PREVIEW_LABEL", "rollout")

    class RolloutPreview(rp.Preview):
        def _compose(self):
            canvas = np.zeros((self.h, self.w, 3), np.uint8)
            names = list(self.cams)
            top = rp.even((self.h - self.big_h) / 2)
            tiles = [(names[0], 0, top, self.big_w, self.big_h)]
            tiles += [(n, self.big_w, i * self.small_h, self.small_w,
                       self.small_h) for i, n in enumerate(names[1:])]
            for name, x, y, w, h in tiles:
                canvas[y:y + h, x:x + w] = self._frame(self.cams[name], w, h)
                cv2.putText(canvas, name, (x + 10, y + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            hz = 0.0
            if len(ticks) > 1 and time.time() - ticks[-1] < 0.5:
                hz = (len(ticks) - 1) / (ticks[-1] - ticks[0])
            colour = (0, 255, 0) if hz >= 27 else (255, 60, 60)
            text = f"{label}   loop {hz:4.1f} Hz" if hz else \
                f"{label}   robot idle"
            cv2.putText(canvas, text, (10, self.h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
            return canvas

        def run(self):
            self._take_focus = lambda: None
            super().run()

    send = robot.send_action

    def send_action(action, *a, **kw):
        ticks.append(time.time())
        return send(action, *a, **kw)

    robot.send_action = send_action
    if robot.cameras:
        RolloutPreview(robot.cameras,
                       int(os.environ.get("PREVIEW_W", 1600)),
                       int(os.environ.get("PREVIEW_FPS", 15))).start()


def patch_preview():
    from lerobot.rollout import context
    make = context.make_robot_from_config

    def make_robot_with_preview(cfg):
        robot = make(cfg)
        start_preview(robot)
        return robot

    context.make_robot_from_config = make_robot_with_preview


if __name__ == "__main__":
    patch_diffusion()
    patch_prepare()
    if int(os.environ.get("SMOOTH", "0")) > 1:
        patch_smooth(int(os.environ["SMOOTH"]) // 2 * 2 + 1)
    if os.environ.get("PREVIEW", "1") != "0":
        patch_preview()
    from lerobot.scripts.lerobot_rollout import main
    main()
