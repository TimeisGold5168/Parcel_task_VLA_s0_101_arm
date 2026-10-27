"""Predict one full B-spline segment with an ACT-BSP or Diffusion-BSP.

LeRobot's diffusion drops the first n_obs_steps - 1 rows of its horizon
at inference; a B-spline segment needs every row, so diffusion sampling
is called directly here.
"""

import torch

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@torch.no_grad()
def predict_segment(policy, obs, noise=None):
    """obs: normalized observation batch. Returns (B, rows, 1 + dim).

    noise: diffusion only, (B, rows, 1 + dim) start noise; None = random.
    """
    if policy.config.type == "act":
        return policy.predict_action_chunk(obs)
    batch = dict(obs)
    feats = policy.config.image_features
    n_obs = policy.config.n_obs_steps
    # one live frame: repeat it to fill the history (LeRobot does the
    # same at the start of an episode)
    for key in feats:
        if batch[key].ndim == 4:
            batch[key] = batch[key].unsqueeze(1).expand(
                -1, n_obs, -1, -1, -1)
    if batch[OBS_STATE].ndim == 2:
        batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1).expand(
            -1, n_obs, -1)
    if feats:
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in feats], dim=-4)
    model = policy.diffusion
    cond = model._prepare_global_conditioning(batch)
    return model.conditional_sample(batch[OBS_STATE].shape[0],
                                    global_cond=cond, noise=noise)
