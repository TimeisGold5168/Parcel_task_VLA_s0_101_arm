"""Faster SmolVLA denoising: read-only prefix K/V cache + torch.compile.

One SmolVLA plan runs the 10 flow-matching steps as about 1,200 small
kernels each; the CPU launches them slower than the GPU runs them.
torch.compile fuses them, and mode "reduce-overhead" also replays the
step as one CUDA graph.

The LeRobot DynamicCache does not work in a CUDA graph: the
self-attention layers append the suffix K/V and denoise_step crops it
again, so the cache changes inside the graph. StaticPrefixCache keeps
the prefix K/V in fixed buffers that the step only reads.

  from parcel_vla.inference_improvement.fast_denoise import enable
  enable(policy.model, "default")   # or "reduce-overhead"
"""

import torch


class _KV:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class StaticPrefixCache:
    """Read-only prefix K/V in fixed buffers (CUDA graph safe).

    Same interface as the DynamicCache calls in smolvlm_with_expert:
    update() returns [prefix; suffix] but stores nothing, so crop() has
    nothing to undo.
    """

    def __init__(self):
        self.layers, self.src = None, None

    def load(self, dyn):
        if self.layers is None:
            self.layers = []
            for lay in dyn.layers:
                kv = _KV(torch.empty_like(lay.keys).contiguous(),
                         torch.empty_like(lay.values).contiguous())
                torch._dynamo.mark_static_address(kv.keys)
                torch._dynamo.mark_static_address(kv.values)
                self.layers.append(kv)
        for kv, lay in zip(self.layers, dyn.layers):
            kv.keys.copy_(lay.keys)
            kv.values.copy_(lay.values)
        self.src = dyn

    def update(self, keys, values, layer_idx):
        kv = self.layers[layer_idx]
        return (torch.cat([kv.keys, keys], dim=2),
                torch.cat([kv.values, values], dim=2))

    def crop(self, max_length):
        pass


def enable(core, mode="default"):
    """Patch core.denoise_step (core = SmolVLA policy.model).

    mode: "none" (cache only), "default" (fused kernels) or
    "reduce-overhead" (fused kernels + CUDA graph). Call under
    torch.inference_mode(), as the BSP executor does. The first plan
    compiles (4-20 s): plan once before the robot moves.
    """
    cache = StaticPrefixCache()
    step = core.denoise_step
    if mode != "none":
        step = torch.compile(step, mode=mode)

    def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        if cache.src is not past_key_values:  # new plan: new prefix
            cache.load(past_key_values)
        torch.compiler.cudagraph_mark_step_begin()
        return step(prefix_pad_masks, cache, x_t, timestep).clone()

    core.denoise_step = denoise_step
