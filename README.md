# H3 Phase 2+

Controls MiniMax H3 **Two Phases** generation (phase 1 at reduced resolution, latent upscale, phase 2 refine) without editing WanGP files. The folder matches `/plugins/wan2gp-*/` in WanGP's `.gitignore`, so updates leave it alone.

## What it changes
| | WanGP built-in | H3 Phase 2+ (defaults) |
|---|---|---|
| Phase 1 size | output / 2 | output / *draft scale* (default 2.0, adjustable 1.25-2.0) |
| Phase 2 steps | 3 | 4 |
| Start noise | 0.9035 (slider min 0.7) | 0.80, adjustable down to 0.30 |
| Schedule | 0.90 → 0.63 → 0.32 → 0 | ComfyUI "simple" at shift 12: 0.80 → 0.735 → 0.632 → 0.444 → 0 |
| Latent upscale | 16-latent-frame chunks | Whole clip in one pass (cross-faded segments if out of VRAM) |
| Phase 2 LoRA | LightX2V FL2V Turbo v0.1, forced to 1.0 | Any LoRA in `loras/minimax_h3` (default `lightx2v_hybrid-4to8step-Turbo_r48` at 0.7), or None |

Defaults mirror the ComfyUI workflow *Minimax H3 Cinematic Version* (stage 2: BasicScheduler simple, 4 steps, denoise 0.25).

## Distorted or blurry faces: bigger draft
Faces are shaped in phase 1. At a 640p output, WanGP's phase 1 is only 576x320 (about 0.18 MP), so a face in a medium or wide shot gets a few dozen pixels and comes out malformed; phase 2 mostly adds detail on top of that shape.

Press **Load 'bigger draft' settings**, then **Save**. It sets:
- **Draft scale 1.5**: phase 1 at output / 1.5 (1152x640 -> 768x416, 1280x720 -> 864x480, about 0.3-0.4 MP).
- **2 steps from start noise 0.63**: a light refine that keeps the better-formed phase 1 (ComfyUI beta scheduler, denoise 0.2 at shift 12).

Your LoRA choice is left as it is. Phase 1 gets slower (about 1.8x the pixels of the default); phase 2 gets cheaper (2 steps instead of 3-4, smaller upscale). If faces still drift in phase 2, lower start noise; if they stay soft, raise it.

The idea is the same two-pass method used in ComfyUI H3 workflows that render a ~0.4 MP draft and refine it after the 3D latent upscaler.

## Flashes every ~2.3 seconds: single-pass upscale
WanGP upscales the phase-1 latent in chunks of 16 latent frames (about 2.3 s each) to save VRAM. The upscaler's GroupNorm layers compute their statistics over each chunk, so neighbouring chunks come out with slightly different brightness or colour. Phase 2 keeps part of the upscaled latent, so each join can survive as a one- or two-frame flash at fixed times (about 2.3 s, 4.5 s, 6.8 s, ...) whatever the scene.

**Single-pass latent upscale** (on by default) runs the whole clip through the upscaler at once, so there are no joins. It needs more VRAM than the chunks (roughly 5-6 GB for a 1920x1088 output of about 14 s; less for smaller or shorter outputs). If the GPU runs out of memory, the plugin prints a line in the console and falls back to segments for that generation, but with 4 frames of context on each side (WanGP uses 2) and a linear cross-fade where they overlap instead of a hard cut, so any difference between segments fades in over a few frames instead of flashing. It works whether or not the phase-2 overrides above are enabled.

The plugin's lower start noise (0.80 vs WanGP's 0.9035) keeps more of the upscaled latent in phase 2, which is why the joins were more visible with it enabled.

Credit: the diagnosis, the fix and its render tests (1920x1088, 345 frames, RTX 4090: no flash at any former join) are by **MOUGE**.

## Install
1. In WanGP open the **Plugins** tab, paste `https://github.com/Arturogpj/wan2gp-h3-phase2-plus` into the install-from-URL box and install.
2. Enable `wan2gp-h3-phase2-plus` → Save → restart WanGP.
3. Download the default phase-2 LoRA into `loras/minimax_h3`: [lightx2v_hybrid-4to8step-Turbo_r48.safetensors](https://huggingface.co/RunningHubAI/rh-lightx2v-hybrid-4to8step-turbo-r48-lora/resolve/main/lightx2v_hybrid-4to8step-Turbo_r48.safetensors) (944 MB). Or pick any other LoRA in that folder (or None) in the plugin tab.

## Use
1. Make sure the plugin is enabled (see Install).
2. Open the **H3 Phase 2+** tab, adjust (or press **Load 'bigger draft' settings**), press **Save**. Settings apply to the next generation (no restart); they live in `config.json` here.
3. In the main tab pick an H3 model, *Advanced Mode → General → Phases → Two Phases*.
4. To keep look LoRAs active during phase 2, give them two multipliers, e.g. `0.7;0.7`.

The console prints `[H3 Phase 2 Plus] Phase 1 at ... Phase 2: ...` at the start of every generation it affects.

## Not changed
- PDD and VDN models, single-phase generation, and the H3 Face Refiner.
- Audio stays frozen in phase 2 (the ComfyUI workflow lightly re-samples it).
- The learned latent upscaler is scale-conditioned, so any draft scale in the slider range is a valid input for it.
- WanGP's *Phase 2 Noise Level Start* slider is ignored while the plugin is enabled.

## How
`h3_patch.py` wraps `MiniMaxH3Pipeline.get_loras_transformer`, `MiniMaxH3Pipeline.generate` and `update_loras_slists` in `models/minimax_h3/pipeline.py`, and `MiniMaxH3LatentUpscaler.forward` in `models/minimax_h3/latent_upscaler.py`, at startup. If a WanGP update renames those, WanGP prints an "Error in setup_ui" line for this plugin at startup instead of silently ignoring it.
