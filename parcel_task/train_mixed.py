"""lerobot-train for Diffusion with cameras of different sizes.

Diffusion stacks the camera images before it resizes them, so all
cameras must have the same shape. The parcel dataset has a 848x480 top
camera and 640x480 wrist and side cameras. This wrapper resizes each
camera to `--policy.resize_shape` before the stack. Use the same patch
(`patch_diffusion()`) at inference time.

Usage: python train_mixed.py <lerobot-train arguments>
"""
import torch.nn.functional as F

from lerobot.policies.diffusion import configuration_diffusion as cd
from lerobot.policies.diffusion import modeling_diffusion as md


def _resize(batch, keys, shape):
    out = dict(batch)
    for k in keys:
        x = out.get(k)
        if x is None or tuple(x.shape[-2:]) == tuple(shape):
            continue
        lead = x.shape[:-3]
        y = F.interpolate(x.reshape(-1, *x.shape[-3:]), size=tuple(shape),
                          mode="bilinear", align_corners=False)
        out[k] = y.reshape(*lead, *y.shape[-3:])
    return out


def patch_diffusion():
    orig_validate = cd.DiffusionConfig.validate_features

    def validate(self):
        if self.resize_shape is None:
            return orig_validate(self)
        if not self.image_features and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the "
                             "environment state among the inputs.")

    cd.DiffusionConfig.validate_features = validate

    for name in ("forward", "select_action", "predict_action_chunk"):
        orig = getattr(md.DiffusionPolicy, name)

        def wrapped(self, batch, *a, _orig=orig, **kw):
            if self.config.resize_shape is not None:
                batch = _resize(batch, self.config.image_features,
                                self.config.resize_shape)
            return _orig(self, batch, *a, **kw)

        setattr(md.DiffusionPolicy, name, wrapped)


if __name__ == "__main__":
    patch_diffusion()
    from lerobot.scripts.lerobot_train import main
    main()
