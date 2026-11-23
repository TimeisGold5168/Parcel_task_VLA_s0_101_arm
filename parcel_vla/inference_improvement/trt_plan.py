"""TensorRT plans for Diffusion-BSP and SmolVLA-BSP (item 10).

The Python caller (BSP executor, scheduler loop, B-spline) does not
change. Only the network parts run as TensorRT engines:

- Diffusion: the image encoder + state (global condition) and the
  1D U-Net (called once per DDIM step).
- SmolVLA: the SigLIP image encoder (once per camera) and the prefix
  pass that fills the K/V cache. The denoising steps stay in PyTorch
  (parcel_vla/inference_improvement/fast_denoise.py can still compile them).

TensorRT 11 builds strongly typed networks only, so the precision is
the dtype of the exported ONNX graph: fp16 / fp32 for Diffusion (the
PyTorch model is fp32), bf16 for SmolVLA (the VLM runs in bf16).

  python -m parcel_vla.inference_improvement.trt_plan build diffusion_bsp_b \
    --dtype fp16
  python -m parcel_vla.inference_improvement.trt_plan build smolvla_bsp_b

  from parcel_vla.inference_improvement.trt_plan import enable
  enable(policy, "diffusion_bsp_b", "fp16")  # after policy.to("cuda")

Engines: engines/<model>_<part>_<dtype>.engine next to this script (one
GPU type).
"""

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

from parcel_vla import paths
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

HERE = Path(__file__).parent
ENGINES = HERE / "engines"
TRAIN = paths.TRAIN
CKPTS = {"diffusion_bsp_b": "diffusion_bsp_parcel_b_40k",
         "smolvla_bsp_b": "smolvla_bsp_parcel_b_40k"}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16,
          "bf16": torch.bfloat16}
TRT2TORCH = {trt.float32: torch.float32, trt.float16: torch.float16,
             trt.bfloat16: torch.bfloat16, trt.int64: torch.int64,
             trt.int32: torch.int32, trt.bool: torch.bool}
LOG = trt.Logger(trt.Logger.WARNING)


def engine_path(model, part, dtype):
    return ENGINES / f"{model}_{part}_{dtype}.engine"


class TRTModule(nn.Module):
    """One engine with fixed shapes. Inputs and outputs are CUDA tensors.

    The outputs are buffers of this module: the next call writes over
    them, so the caller must use (or copy) them first.
    """

    def __init__(self, path):
        super().__init__()
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(
            Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.ins, self.outs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dt = TRT2TORCH[self.engine.get_tensor_dtype(name)]
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.ins[name] = (shape, dt)
            else:
                buf = torch.empty(shape, dtype=dt, device="cuda")
                self.outs[name] = buf
                self.ctx.set_tensor_address(name, buf.data_ptr())

    def forward(self, **feeds):
        keep = []
        for name, (shape, dt) in self.ins.items():
            t = feeds[name].to(dt).contiguous()
            if tuple(t.shape) != shape:
                raise ValueError(f"{name}: shape {tuple(t.shape)}, "
                                 f"engine {shape}")
            keep.append(t)
            self.ctx.set_tensor_address(name, t.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self.ctx.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute failed")
        return self.outs


def build_engine(onnx_path, out_path, workspace_gb=1.0):
    builder = trt.Builder(LOG)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, LOG)
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * 2**30))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"engine build failed: {onnx_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))


# ---------------------------------------------------------------- Diffusion

class DiffusionCond(nn.Module):
    """images (1, s, n, 3, H, W), state (1, s, dim) -> global condition."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, images, state):
        return self.m._prepare_global_conditioning(
            {OBS_STATE: state, OBS_IMAGES: images})


class DiffusionUnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, t, cond):
        return self.m.unet(x, t, global_cond=cond)


def diffusion_examples(policy):
    cfg = policy.config
    s, n = cfg.n_obs_steps, len(cfg.image_features)
    h, w = cfg.resize_shape
    dim = cfg.robot_state_feature.shape[0]
    images = torch.rand(1, s, n, 3, h, w)
    state = torch.randn(1, s, dim)
    dev, dt = _dev(policy), next(policy.parameters()).dtype
    with torch.inference_mode():
        cond_dim = policy.diffusion._prepare_global_conditioning(
            {OBS_STATE: state.to(dev, dt),
             OBS_IMAGES: images.to(dev, dt)}).shape[-1]
    x = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0])
    t = torch.tensor([5], dtype=torch.long)
    return {"cond": (images, state),
            "unet": (x, t, torch.randn(1, cond_dim))}


class _TRTCond(nn.Module):
    def __init__(self, eng):
        super().__init__()
        self.eng = eng

    def forward(self, batch):
        out = self.eng(images=batch[OBS_IMAGES], state=batch[OBS_STATE])
        return out["out"].float()


class _TRTUnet(nn.Module):
    def __init__(self, eng):
        super().__init__()
        self.eng = eng

    def forward(self, x, t, global_cond=None):
        return self.eng(x=x, t=t, cond=global_cond)["out"].float()


def enable_diffusion(policy, model, dtype):
    d = policy.diffusion
    cond = _TRTCond(TRTModule(engine_path(model, "cond", dtype)))
    unet = _TRTUnet(TRTModule(engine_path(model, "unet", dtype)))
    d.unet = unet
    # a method, not a submodule: it must go in the instance dict
    object.__setattr__(d, "_prepare_global_conditioning", cond)


# ---------------------------------------------------------------- SmolVLA

class SmolImage(nn.Module):
    def __init__(self, vwe):
        super().__init__()
        self.vwe = vwe

    def forward(self, image):
        return self.vwe.embed_image(image)


class SmolPrefix(nn.Module):
    """Prefix pass: embeddings, 2D mask, positions -> K, V of all layers.

    Out: keys, values (layers, 1, heads, len, head_dim), as stored by
    the DynamicCache during the LeRobot prefix pass.
    """

    def __init__(self, vwe):
        super().__init__()
        self.vwe = vwe

    def forward(self, embs, mask, pos):
        _, cache = self.vwe.forward(attention_mask=mask, position_ids=pos,
                                    past_key_values=None,
                                    inputs_embeds=[embs, None],
                                    use_cache=True)
        return (torch.stack([lay.keys for lay in cache.layers]),
                torch.stack([lay.values for lay in cache.layers]))


def smolvla_examples(policy, dtype):
    core = policy.model
    cfg = policy.config
    h, w = cfg.resize_imgs_with_padding
    dev = _dev(policy)
    image = torch.rand(1, 3, h, w, device=dev) * 2 - 1
    n_img = len(cfg.image_features)
    emb = core.vlm_with_expert.embed_image(image)
    n_tok = n_img * emb.shape[1] + cfg.tokenizer_max_length + 1
    hid = emb.shape[-1]
    embs = torch.randn(1, n_tok, hid, device=dev, dtype=dtype)
    mask = torch.ones(1, n_tok, n_tok, dtype=torch.bool, device=dev)
    mask[:, : n_tok - 1, -1] = False
    pos = torch.arange(n_tok, device=dev)[None]
    return {"image": (image.to(dtype),), "prefix": (embs, mask, pos)}


class _CacheLayer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


def _make_cache(keys, values):
    from transformers import DynamicCache

    cache = DynamicCache()
    for i in range(keys.shape[0]):
        cache.update(keys[i].clone(), values[i].clone(), i)
    return cache


def enable_smolvla(policy, model, dtype="bf16"):
    vwe = policy.model.vlm_with_expert
    img = TRTModule(engine_path(model, "image", dtype))
    pre = TRTModule(engine_path(model, "prefix", dtype))
    vdt = vwe.get_vlm_model().vision_model.dtype
    fwd = vwe.forward

    def embed_image(image):
        return img(image=image)["out"].to(vdt).clone()

    def forward(attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None):
        prefix_only = (past_key_values is None and use_cache
                       and inputs_embeds[1] is None)
        if not prefix_only:
            return fwd(attention_mask=attention_mask,
                       position_ids=position_ids,
                       past_key_values=past_key_values,
                       inputs_embeds=inputs_embeds, use_cache=use_cache)
        out = pre(embs=inputs_embeds[0], mask=attention_mask,
                  pos=position_ids)
        return [None, None], _make_cache(out["keys"], out["values"])

    vwe.embed_image = embed_image
    vwe.forward = forward


# ---------------------------------------------------------------- common

def _dev(policy):
    return next(policy.parameters()).device


def enable(policy, model, dtype):
    """Patch a loaded policy (on CUDA) to plan with the engines."""
    if policy.config.type == "diffusion":
        enable_diffusion(policy, model, dtype)
    else:
        enable_smolvla(policy, model, dtype)


def load_policy(model, device="cuda", ddim_steps=10):
    """The policy as run_bsp_parcel.py loads it (DDIM, bf16 VLM)."""
    from parcel_vla.training.train_mixed import patch_diffusion

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    path = TRAIN / CKPTS[model] / "checkpoints/last/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(path)
    if cfg.type == "diffusion":
        patch_diffusion()
        cfg.noise_scheduler_type = "DDIM"
        cfg.num_inference_steps = ddim_steps
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    if cfg.type == "smolvla":
        vlm = policy.model.vlm_with_expert.vlm
        for prm in vlm.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)
    return policy.to(device).eval(), path


def export(model, dtype, device, out_dir, check_cpu=False):
    """Export the engine parts of one model to ONNX (one file each)."""
    policy, _ = load_policy(model, device)
    tdt = DTYPES[dtype]
    out_dir.mkdir(parents=True, exist_ok=True)
    if policy.config.type == "diffusion":
        policy = policy.to(tdt)
        # the sinusoidal timestep embedding is float32
        lin = policy.diffusion.unet.diffusion_step_encoder[1]
        lin.register_forward_pre_hook(
            lambda m, a: (a[0].to(m.weight.dtype),))
        ex = diffusion_examples(policy)
        parts = {"cond": (DiffusionCond(policy.diffusion),
                          ["images", "state"]),
                 "unet": (DiffusionUnet(policy.diffusion),
                          ["x", "t", "cond"])}
        ex = {k: tuple(a.to(device).to(tdt) if a.is_floating_point()
                       else a.to(device) for a in v)
              for k, v in ex.items()}
        outs = {"cond": ["out"], "unet": ["out"]}
    else:
        if dtype == "fp32":
            policy.model.vlm_with_expert.vlm.float()
        ex = smolvla_examples(policy, tdt)
        vwe = policy.model.vlm_with_expert
        parts = {"image": (SmolImage(vwe), ["image"]),
                 "prefix": (SmolPrefix(vwe), ["embs", "mask", "pos"])}
        outs = {"image": ["out"], "prefix": ["keys", "values"]}
    files = {}
    for part, (mod, names) in parts.items():
        f = out_dir / f"{model}_{part}_{dtype}.onnx"
        with torch.inference_mode():
            ref = mod(*ex[part])
        try:
            torch.onnx.export(mod, ex[part], str(f), input_names=names,
                              output_names=outs[part], opset_version=20,
                              dynamo=False, external_data=True)
        except Exception as err:  # noqa: BLE001
            print(f"  TorchScript export failed ({type(err).__name__}); "
                  "torch.export")
            prog = torch.onnx.export(mod, ex[part], input_names=names,
                                     output_names=outs[part],
                                     opset_version=20, dynamo=True)
            prog.save(str(f), external_data=True)
        files[part] = (f, ex[part], ref)
        print(f"exported {f.name}")
        if check_cpu:
            _check_ort(f, names, ex[part], ref)
    return files


def _check_ort(f, names, args, ref):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    feeds = {n: a.detach().cpu().numpy() for n, a in zip(names, args)}
    got = sess.run(None, feeds)
    refs = ref if isinstance(ref, tuple) else (ref,)
    for g, r in zip(got, refs):
        err = float(abs(torch.from_numpy(g) - r.float().cpu()).max())
        print(f"  ONNX Runtime CPU vs PyTorch: max |diff| {err:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "build"])
    ap.add_argument("model", choices=list(CKPTS))
    ap.add_argument("--dtype", default=None, choices=list(DTYPES),
                    help="default: fp16 (diffusion), bf16 (smolvla)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-cpu", action="store_true",
                    help="run the ONNX files in ONNX Runtime (CPU, fp32)")
    args = ap.parse_args()
    dtype = args.dtype or ("fp16" if "diffusion" in args.model
                           else "bf16")
    onnx_dir = ENGINES / "onnx" / f"{args.model}_{dtype}"
    files = export(args.model, dtype, args.device, onnx_dir,
                   args.check_cpu)
    if args.cmd == "build":
        for part, (f, _, _) in files.items():
            out = engine_path(args.model, part, dtype)
            build_engine(f, out)
            print(f"built {out.name}")
    meta = ENGINES / "engines.json"
    info = json.loads(meta.read_text()) if meta.exists() else {}
    info[f"{args.model}_{dtype}"] = {
        "trt": trt.__version__, "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name()
                if torch.cuda.is_available() else "")}
    ENGINES.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
