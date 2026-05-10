# ControlNet-LLLite for Anima — Installation

## Prerequisites

- Stable Diffusion WebUI Forge Neo (this repo)
- Anima model loaded as the active checkpoint
- A trained ControlNet-LLLite for Anima weight file (`.safetensors` or `.pt`)

---

## Model file placement

Place the weight file in the ControlNet model directory:

```
models/
  ControlNet/
    your_anima_lllite.safetensors
```

The directory is scanned recursively, so subdirectories are fine:

```
models/ControlNet/lllite/your_anima_lllite.safetensors
```

The default path is `models/ControlNet`. To use a different location, pass `--controlnet-dir` at launch:

```bat
:: webui-user.bat
set COMMANDLINE_ARGS=--uv --api --controlnet-dir D:\my-models\ControlNet
```

---

## Detection

Forge identifies Anima LLLite weights automatically by looking for keys containing `lllite_dit` in the state dict. No manual model-type selection is needed.

| Key pattern | Loaded as |
|---|---|
| `lllite_dit_*` | ControlNet-LLLite for Anima |
| `lllite_unet_*` | ControlNet-LLLite for SDXL |

Anima LLLite weights are claimed before the generic SDXL LLLite loader. If an Anima-style weight cannot be matched to the active DiT checkpoint, Forge reports the Anima LLLite load error instead of falling back to the SDXL path.

---

## Usage

1. Load an Anima checkpoint.
2. Open the **ControlNet** panel in the UI.
3. Select your Anima LLLite file from the model dropdown.
4. Upload a control image (edge map, depth map, etc.).
5. Set **Control Weight** (strength) and optionally **Starting/Ending Control Step**.
6. Generate.

The ControlNet LLLite panel is the same UI as for other ControlNet models; no separate panel exists.

---

## How it works (brief)

The weight file encodes a small conditioning network (`_Conditioning1`) and one `LLLiteModuleDiT` per targeted attention/MLP projection. At inference:

1. The conditioning image is passed through `_Conditioning1` → `(B, S, cond_emb_dim)` embeddings.
2. Each targeted `Linear.forward()` inside the Anima DiT is temporarily wrapped to add a FiLM-modulated correction derived from those embeddings.
3. After sampling, all `Linear.forward()` methods are restored to their originals.

The model file stores the following key groups:

| Key prefix | Contents |
|---|---|
| `lllite_conditioning1.*` | Shared image encoder (conv + resblocks + optional ASPP) |
| `lllite_dit_blocks_N_*.{down,mid,cond_to_film,up}.*` | Per-module MLP weights |
| `lllite_dit_blocks_N_*.depth_embed` | Per-module depth embedding (zero-init) |

---

## Troubleshooting

**Model does not appear in the dropdown**
The ControlNet model list is cached at startup. Click Refresh next to the model dropdown, or restart the server.

**Model appears but has no effect**
Verify that an Anima checkpoint is loaded. Anima LLLite weights only work with the Anima DiT architecture. If the active model exposes no matching `SelfCrossAttention` or `GPT2FeedForward` modules, Forge raises a clear `Control-LLLite (Anima) found no target modules` error.

**Shape mismatch / fallback to identity**
For video generation (T > 1 frames), the spatial token count of the conditioning embedding may not match the sequence length of the attention input. The module falls back to the original Linear in that case. Providing a conditioning image resized to match the output spatial resolution reduces the chance of mismatch.

**`depth_embed slices missing` error**
The weight file was saved with a different number of modules than the current model produces. This happens if the weight was trained on a different version of the Anima architecture. Use a weight file that matches the loaded checkpoint.

---

## Local validation

The implementation was validated with three local test weights:

| Weight file | Target | Modules |
|---|---|---|
| `anima-lllite-any-test-like-1-step1000.safetensors` | `self_attn_q_pre` | 28 |
| `anima-lllite-any-test-like-1-step2000.safetensors` | `self_attn_q_pre` | 28 |
| `anima-lllite-any-test-like-3-step00000550.safetensors` | `self_attn_q_pre` | 28 |

Each file loaded with zero missing and zero unexpected keys against the current Anima DiT module names. A runtime txt2img API generation on `waiANIMA_v10.safetensors` also loaded `ControlLLLiteAnimaPatcher` and reported `Loaded Control-LLLite (Anima) (28 modules)`.
