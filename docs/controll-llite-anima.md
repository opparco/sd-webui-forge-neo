# ControlNet-LLLite: SDXL vs Anima Architecture

## Overview

This repo includes ControlNet-LLLite support for SDXL via the `sd_forge_controlllite` built-in extension. Adding Anima (DiT) support requires a separate implementation due to fundamental architectural differences.

- SDXL implementation: `extensions-builtin/sd_forge_controlllite/lib_controllllite/lib_controllllite.py`
- Anima reference: [kohya-ss/ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite/blob/main/control_net_lllite_anima.py)

---

## Comparison

### Hook mechanism

| | SDXL | Anima |
|---|---|---|
| Target model | U-Net (input / middle / output blocks) | DiT (Transformer blocks) |
| Hook style | ComfyUI `attn1_patch` / `attn2_patch` — receives `(q, k, v, extra_options)` and modifies them in-place | `Linear.forward()` is directly replaced via `apply_to()` / `restore()` |
| Module identification | Reconstructs module name from `extra_options["block"]` and `block_index` | `dit.named_modules()` at construction time; filters by class name (`Attention`, `GPT2FeedForward`) |
| Registration | `model.set_model_attn1_patch(patch)` / `set_model_attn2_patch(patch)` | `apply_to()` before sampling, `restore()` after |

### Weight key format

| | SDXL | Anima |
|---|---|---|
| Module keys | `lllite_unet_input_blocks_N_1_transformer_blocks_M_attn1_to_q.*` | `lllite_dit_blocks_N_self_attn_q_proj.*` |
| Conditioning keys | embedded in module keys | `lllite_conditioning1.*` (shared trunk) |
| Depth embedding | none | `{module_name}.depth_embed` (per-module, zero-init) |
| Legacy format | accepted | `lllite_modules.*` prefix is rejected with an error |

### Conditioning trunk (`conditioning1`)

**SDXL** — simple Conv2d stack, depth 1–3:

```
Conv2d(3, cond_emb_dim//2, k=4, s=4)  →  ReLU
  depth 1: Conv2d(→ cond_emb_dim, k=2, s=2)
  depth 2: Conv2d(→ cond_emb_dim, k=4, s=4)
  depth 3: Conv2d(→ cond_emb_dim//2, k=4, s=4) → ReLU → Conv2d(→ cond_emb_dim, k=2, s=2)
```

**Anima** — richer encoder with residual blocks and optional ASPP:

```
Conv2d(3, cond_dim//2, k=4, s=4)  + GroupNorm + SiLU
Conv2d(→ cond_dim//2, k=3, s=1)  + GroupNorm + SiLU
Conv2d(→ cond_dim,    k=4, s=4)  + GroupNorm + SiLU
N × ResBlock(cond_dim)            [optional, n=cond_resblocks]
ASPP(cond_dim, dilations)         [optional]
Conv2d(→ cond_emb_dim, k=1)
reshape b,c,h,w → b,h*w,c
LayerNorm(cond_emb_dim)
```

Output shape: `(B, S, cond_emb_dim)` where S = spatial tokens.

### LLLite module (per target Linear)

**SDXL** (`LLLiteModule`) — supports both Conv2d and Linear:

```
cx  = conditioning1(cond_image)          # (B, cond_emb_dim, h, w) or (B, S, cond_emb_dim)
h   = down(x)                            # Linear or Conv2d
mid_input = cat([cx, h], dim=channel)
cx  = mid(mid_input)                     # ReLU
out = up(cx) * multiplier
return x + out
```

**Anima** (`LLLiteModuleDiT`) — Linear-only, FiLM modulation, per-module depth embedding:

```
cond_local = cond_emb + depth_embeds[layer_idx]   # depth embed: zero-init → no-op at init
h          = silu(down(x))
gamma, beta = cond_to_film(cond_local).chunk(2)   # zero-init → identity at init
m          = mid(cat([cond_local, h]))
m          = silu(m * (1 + gamma) + beta)          # FiLM
out        = up(m) * multiplier
return org_forward(x + out)                        # wraps the original Linear
```

Key differences:
- **FiLM** (Feature-wise Linear Modulation) instead of simple concatenation into `mid`
- **`depth_embeds`** — a shared `nn.Parameter` of shape `(n_modules, cond_emb_dim)`, zero-initialized
- **Wraps `org_forward`**: the correction `out` is added to `x` *before* the original Linear, not after
- **5D support**: MLP path receives `(B, T, H, W, D)`; reshaped to `(B, T*H*W, D)` for the LLLite path then restored
- **dtype safety**: `x` and `cond_emb` are cast to the LLLite parameter dtype before computation, then the correction is cast back

### Target layer selection

**SDXL** — always patches `attn1` and `attn2` for all matching blocks; `to_q`, `to_k`, `to_v` per attention.

**Anima** — configurable via `target_layers` string:

| Preset | Atomics covered |
|---|---|
| `self_attn_q` | self-attention Q only |
| `self_attn_qkv` | self-attention Q, K, V |
| `self_attn_qkv_cross_q` | self-attention Q/K/V + cross-attention Q |
| comma-separated atomics | any combination of `self_attn_q_pre`, `self_attn_kv_pre`, `cross_attn_q_pre`, `mlp_fc1_pre` |

Cross-attention K/V are intentionally excluded (they live in text-embedding space).

### Step range

Both versions support `start_percent` / `end_percent` to limit which denoising steps the control is active.

**SDXL** — tracked via `current_step` counter on each `LLLiteModule`; returns `zeros_like(x)` outside the range.

**Anima** — same `current_step` counter approach as SDXL, implemented in `LLLiteModuleDiT`. `set_step_range()` is called in `process_before_every_sampling` to reset the counter and set the active window. Outside the range, `forward()` falls through to `org_forward` with no modification.

### Control Mode

**SDXL / standard ControlNet** — `positive_advanced_weighting` and `negative_advanced_weighting` are applied inside `compute_controlnet_weighting` (`backend/patcher/controlnet.py`), which is called from `ControlBase.control_merge()` → `ControlNet.get_control()`. This runs every sampling step via the `controlnet_linked_list` on the UNet patcher:

```
sampling_function
  control = unet_patcher.controlnet_linked_list   # registered via add_patched_controlnet()
  control.get_control(...)
    control_merge()
      compute_controlnet_weighting()   # applies positive/negative weighting here
```

The weighting dict keys (`"input"`, `"middle"`, `"output"`) map directly to UNet block types, and each value list maps to per-layer residual tensors produced by the ControlNet model.

**Anima** — `ControlLLLiteAnimaPatcher` never calls `add_patched_controlnet()`. The LLLite net patches `Linear.forward()` directly and is not part of the ControlNet residual pipeline. As a result, `controlnet_linked_list` is not set and `compute_controlnet_weighting` is never invoked. The UNet block/layer hierarchy that `soft_weighting` / `zero_weighting` targets does not exist in a DiT.

**Control Mode is therefore not implemented for Anima LLLite.** The `positive_advanced_weighting` / `negative_advanced_weighting` fields are set on the patcher by the ControlNet extension but are intentionally ignored.

---

## Forge Neo Integration Status

The Anima ControlNet-LLLite path is implemented in Forge Neo and validated with local `lllite_dit_*` weights. The patcher is registered before the generic SDXL LLLite patcher, builds the DiT hooks lazily during sampling, restores patched `Linear.forward()` methods after sampling, and raises a clear error if no target modules are found.

### New files

```
extensions-builtin/sd_forge_controlllite/
  lib_controllllite/
    lib_controllllite_anima.py   # ControlNetLLLiteDiT + helpers
  scripts/
    forge_controllllite.py       # add ControlLLLiteAnimaPatcher
```

### `ControlLLLiteAnimaPatcher` flow

```python
class ControlLLLiteAnimaPatcher(ControlModelPatcher):

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if any('lllite_dit' in k for k in state_dict):
            return ControlLLLiteAnimaPatcher(state_dict)
        return None

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self._lllite_net = None   # built lazily on first sample

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        dit  = unet.model.diffusion_model

        if self._lllite_net is None:
            cfg  = infer_anima_config(self.state_dict)
            self._lllite_net = ControlNetLLLiteDiT(dit, **cfg)
            load_lllite_weights(self._lllite_net, self.state_dict)

        cond_image = cond * 2.0 - 1.0          # [0,1] → [-1,1]
        self._lllite_net.set_cond_image(cond_image)
        self._lllite_net.set_multiplier(self.strength)
        self._lllite_net.apply_to()
        unet.add_extra_torch_module_during_sampling(self._lllite_net)

        process.sd_model.forge_objects.unet = unet

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        if self._lllite_net is not None:
            self._lllite_net.restore()
            self._lllite_net.clear_cond_image()
```

### Config inference from state_dict

`ControlNetLLLiteDiT.__init__` needs `cond_emb_dim`, `mlp_dim`, `cond_dim`, `cond_resblocks`, `use_aspp`, and `target_layers`. These can be read back from the saved weights:

| Parameter | Source key / shape |
|---|---|
| `cond_emb_dim` | `lllite_conditioning1.proj.weight` shape[0] |
| `cond_dim` | `lllite_conditioning1.conv1.weight` shape[0] × 2 |
| `mlp_dim` | any `{module}.down.weight` shape[0] |
| `cond_resblocks` | max index in `lllite_conditioning1.resblocks.*` + 1 |
| `use_aspp` | presence of `lllite_conditioning1.aspp.*` keys |
| `target_layers` | module name suffixes: `q_proj` → `self_attn_q_pre`, `k/v_proj` → `self_attn_kv_pre`, `cross_attn*q_proj` → `cross_attn_q_pre`, `layer1` → `mlp_fc1_pre` |

### SDXL vs Anima dispatch

Update `ControlLLLitePatcher.try_build_from_state_dict` to exclude Anima weights:

```python
@staticmethod
def try_build_from_state_dict(state_dict, ckpt_path):
    if any('lllite_dit' in k for k in state_dict):
        return None   # handled by ControlLLLiteAnimaPatcher
    if not any('lllite' in k for k in state_dict):
        return None
    return ControlLLLitePatcher(state_dict)
```

Register `ControlLLLiteAnimaPatcher` before `ControlLLLitePatcher` so it is checked first.

### Validation

- Local test weights with `lllite_dit_blocks_N_self_attn_q_proj.*` keys map to 28 `blocks.N.self_attn.q_proj` modules.
- The inferred config for those weights is `cond_emb_dim=32`, `cond_dim=64`, `mlp_dim=64`, `cond_resblocks=6`, `target_layers=self_attn_q_pre`, `use_aspp=False`.
- Runtime txt2img generation with `waiANIMA_v10.safetensors` and `anima-lllite-any-test-like-1-step1000.safetensors` successfully loads `ControlLLLiteAnimaPatcher` and reports `Loaded Control-LLLite (Anima) (28 modules)`.
- A fixed-seed comparison between Control Weight `0.0` and `1.0` produced different output pixels, confirming the hook path affects sampling.

### Caveats

- `ControlNetLLLiteDiT.__init__` requires a live reference to the DiT model to discover target `Linear` modules. It must be built lazily inside `process_before_every_sampling`, not in `__init__` of the patcher.
- `apply_to()` has an idempotency guard (`if self.org_forward is None`), so double-wrapping is safe if `restore()` is always called in `process_after_every_sampling`.
- The Anima model structure (`dit.named_modules()`) must expose `SelfCrossAttention` modules with an `is_SelfAttn` attribute and `GPT2FeedForward.layer1` modules for the module-discovery logic to work.
