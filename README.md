# H3 Phase 2+

Controls the second phase of MiniMax H3 **Two Phases** generation (phase 1 at half resolution, latent upscale, phase 2 refine) without editing WanGP files. The folder matches `/plugins/wan2gp-*/` in WanGP's `.gitignore`, so updates leave it alone.

## What it changes
| | WanGP built-in | H3 Phase 2+ (defaults) |
|---|---|---|
| Phase 2 steps | 3 | 4 |
| Start noise | 0.9035 (slider min 0.7) | 0.80, adjustable down to 0.30 |
| Schedule | 0.90 → 0.63 → 0.32 → 0 | ComfyUI "simple" at shift 12: 0.80 → 0.735 → 0.632 → 0.444 → 0 |
| Phase 2 LoRA | LightX2V FL2V Turbo v0.1, forced to 1.0 | Any LoRA in `loras/minimax_h3` (default `lightx2v_hybrid-4to8step-Turbo_r48` at 0.7), or None |

Defaults mirror the ComfyUI workflow *Minimax H3 Cinematic Version* (stage 2: BasicScheduler simple, 4 steps, denoise 0.25).

## Install
1. In WanGP open the **Plugins** tab, paste `https://github.com/Arturogpj/wan2gp-h3-phase2-plus` into the install-from-URL box and install.
2. Enable `wan2gp-h3-phase2-plus` → Save → restart WanGP.
3. Download the default phase-2 LoRA into `loras/minimax_h3`: [lightx2v_hybrid-4to8step-Turbo_r48.safetensors](https://huggingface.co/RunningHubAI/rh-lightx2v-hybrid-4to8step-turbo-r48-lora/resolve/main/lightx2v_hybrid-4to8step-Turbo_r48.safetensors) (944 MB). Or pick any other LoRA in that folder (or None) in the plugin tab.

## Use
1. Make sure the plugin is enabled (see Install).
2. Open the **H3 Phase 2+** tab, adjust, press **Save**. Settings apply to the next generation (no restart); they live in `config.json` here.
3. In the main tab pick an H3 model, *Advanced Mode → General → Phases → Two Phases*.
4. To keep look LoRAs active during phase 2, give them two multipliers, e.g. `0.7;0.7`.

The console prints `[H3 Phase 2 Plus] Phase 2: ...` at the start of every generation it affects.

## Not changed
- PDD and VDN models, single-phase generation, and the H3 Face Refiner.
- Audio stays frozen in phase 2 (the ComfyUI workflow lightly re-samples it).
- The upscale ratio stays 2× per side (the learned latent upscaler is 2×).
- WanGP's *Phase 2 Noise Level Start* slider is ignored while the plugin is enabled.

## How
`h3_patch.py` wraps `MiniMaxH3Pipeline.get_loras_transformer`, `MiniMaxH3Pipeline.generate` and `update_loras_slists` in `models/minimax_h3/pipeline.py` at startup. If a WanGP update renames those, WanGP prints an "Error in setup_ui" line for this plugin at startup instead of silently ignoring it.
