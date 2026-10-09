"""Runtime overrides for MiniMax H3 two-phase (latent upscale) generation.

Nothing in WanGP's own files is modified. At plugin load we wrap three names in
``models.minimax_h3.pipeline`` and the latent upscaler's ``forward``; every value is
re-read from ``config.json`` at the start of each generation, so changes in the plugin
tab apply without a restart.

- ``MiniMaxH3Pipeline.get_loras_transformer``: inject the chosen phase-2 LoRA
  (phase 1 multiplier 0) instead of the forced LightX2V v0.1 Turbo LoRA.
- ``MiniMaxH3Pipeline.generate``: swap in a ComfyUI-style phase-2 sigma schedule
  (start noise, step count, shift) and the phase-1 draft scale for the duration of one call.
- ``update_loras_slists``: when phase 2 starts, set the chosen LoRA's multiplier
  and mute any other Turbo LoRA so accelerators never stack.
- ``MiniMaxH3LatentUpscaler.forward``: run the whole clip in one pass instead of
  16-latent-frame chunks. Its GroupNorm layers normalize over time, so each chunk got
  its own statistics and the joins showed up as one- or two-frame flashes about every
  2.3 s. If the GPU runs out of memory, falls back to cross-faded segments
  (``crossfaded_upscale``). Diagnosis and fix design by MOUGE.
"""

import json
import os
import threading

import torch

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PLUGIN_DIR, "config.json")
NO_LORA = "None"

# Defaults mirror Junkun Studio's "Minimax H3 Cinematic Version" stage 2:
# BasicScheduler simple / 4 steps / denoise 0.25 at shift 12 -> starts at sigma 0.80,
# with lightx2v_hybrid-4to8step-Turbo_r48 at 0.70 in the model stack.
DEFAULTS = {
    "enabled": True,
    "lora": "lightx2v_hybrid-4to8step-Turbo_r48.safetensors",
    "lora_strength": 0.7,
    "steps": 4,
    "start_noise": 0.80,
    "shift": 12.0,
    # Phase 1 renders at output size / draft_scale. 2.0 is WanGP's built-in ratio.
    "draft_scale": 2.0,
    # Upscale the whole clip at once (no chunk seams); chunked fallback on out-of-memory.
    "single_pass_upscale": True,
}
DRAFT_SCALE_MIN, DRAFT_SCALE_MAX = 1.25, 2.0

_lock = threading.RLock()
_patched = False
_run = {"active": False, "lora": None, "strength": 0.0, "loras_selected": None}
_model_allowed = {"value": True}


def load_config():
    config = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as reader:
            config.update(json.load(reader))
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"[H3 Phase 2 Plus] Could not read {CONFIG_PATH}, using defaults: {error}")
    config["enabled"] = bool(config["enabled"])
    config["single_pass_upscale"] = bool(config["single_pass_upscale"])
    config["lora"] = str(config["lora"] or NO_LORA)
    config["lora_strength"] = float(config["lora_strength"])
    config["steps"] = max(1, min(20, int(config["steps"])))
    config["start_noise"] = max(0.05, min(0.999, float(config["start_noise"])))
    config["shift"] = max(1.0, float(config["shift"]))
    config["draft_scale"] = max(DRAFT_SCALE_MIN, min(DRAFT_SCALE_MAX, float(config["draft_scale"])))
    return config


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as writer:
        json.dump(config, writer, indent=4)


def build_sigmas(start_noise, steps, shift):
    """ComfyUI BasicScheduler("simple", steps, denoise) equivalent, driven by the start sigma.

    The start sigma is mapped back to its unshifted time t0; the remaining steps are spaced
    evenly in unshifted time and re-shifted. With shift 12, start 0.80 and 4 steps this gives
    0.80, 0.735, 0.632, 0.444, 0 -- the schedule of denoise 0.25 in ComfyUI.
    """
    t0 = start_noise / (shift - (shift - 1.0) * start_noise)
    sigmas = []
    for index in range(steps + 1):
        t = t0 * (steps - index) / steps
        sigmas.append(shift * t / (1.0 + (shift - 1.0) * t))
    sigmas[0], sigmas[-1] = float(start_noise), 0.0
    return tuple(sigmas)


def draft_size(width, height, draft_scale):
    """Phase-1 size WanGP renders for a given output size (same rounding as pipeline.py)."""
    return max(32, round(width / draft_scale / 32) * 32), max(32, round(height / draft_scale / 32) * 32)


def describe(config):
    upscale = (" Latent upscale: single pass (cross-faded segments if out of memory)." if config.get("single_pass_upscale", True)
               else " Latent upscale: WanGP's 16-frame chunks.")
    if not config["enabled"]:
        return "Disabled: WanGP's built-in phase 2 is used (3 steps, forced LightX2V v0.1 Turbo at 1.0)." + upscale
    sigmas = build_sigmas(config["start_noise"], config["steps"], config["shift"])
    lora = "no LoRA" if config["lora"] == NO_LORA else f"{config['lora']} @ {config['lora_strength']:g}"
    examples = ", ".join("{}x{} -> {}x{}".format(w, h, *draft_size(w, h, config["draft_scale"])) for w, h in ((1152, 640), (1280, 720)))
    return (f"Phase 1 at output / {config['draft_scale']:g} (e.g. {examples}). "
            f"Phase 2: {config['steps']} steps, sigmas {' -> '.join(f'{s:.3f}' for s in sigmas)}, {lora}." + upscale)


def crossfaded_upscale(model, latent, scale, target_size=None, abort_callback=None, progress_callback=None):
    """Out-of-memory fallback for the latent upscaler (design by MOUGE).

    Same segments as WanGP, but each one gets chunk/4 frames of context on both sides
    (WanGP uses 2) and neighbours are blended with a linear cross-fade over their overlap
    instead of a hard cut, so per-segment GroupNorm differences fade in instead of flashing.
    """
    from models.minimax_h3.latent_upscaler import _check_abort

    frames = latent.shape[2]
    if target_size is None:
        target_size = (frames, int(round(latent.shape[-2] * float(scale))), int(round(latent.shape[-1] * float(scale))))
    _, height, width = (int(size) for size in target_size)
    chunk = model.temporal_chunk_size
    context = max(1, chunk // 4)
    starts = range(0, frames, chunk)
    total_steps = len(starts) * (len(model.in_blocks) + len(model.out_blocks) + 3)
    step = 0

    def advance():
        nonlocal step
        if callable(progress_callback):
            progress_callback("Latent network", step, total_steps)
        step += 1

    device = latent.device
    fade = (torch.arange(2 * context, dtype=torch.float32, device=device) + 0.5) / (2 * context)
    output = torch.zeros(latent.shape[0], latent.shape[1], frames, height, width, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(frames, dtype=torch.float32, device=device)
    for start in starts:
        _check_abort(abort_callback)
        low, high = max(0, start - context), min(frames, start + chunk + context)
        segment = model._forward_segment(latent[:, :, low:high], scale, (high - low, height, width), abort_callback, advance)
        weights = torch.ones(high - low, dtype=torch.float32, device=device)
        ramp = min(2 * context, high - low)
        if low > 0:  # fade in over the overlap with the previous segment
            weights[:ramp] = fade[:ramp]
        if high < frames:  # fade out over the overlap with the next segment
            weights[-ramp:] = torch.minimum(weights[-ramp:], fade.flip(0)[-ramp:])
        output[:, :, low:high] += segment.float() * weights.view(1, 1, -1, 1, 1)
        weight_sum[low:high] += weights
        segment = None
    return output.div_(weight_sum.view(1, 1, -1, 1, 1)).to(latent.dtype)


def _basename(lora):
    return os.path.basename(str(lora).split("|", 1)[0]).lower()


def apply_patches():
    global _patched
    with _lock:
        if _patched:
            return
        from models.minimax_h3 import pipeline
        from models.minimax_h3.latent_upscaler import MiniMaxH3LatentUpscaler

        original_get_loras = pipeline.MiniMaxH3Pipeline.get_loras_transformer
        original_generate = pipeline.MiniMaxH3Pipeline.generate
        original_update_slists = pipeline.update_loras_slists
        original_upscale = MiniMaxH3LatentUpscaler.forward
        out_of_memory = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

        def get_loras_transformer(self, *args, **kwargs):
            model_def = kwargs.get("model_def") or {}
            _model_allowed["value"] = not (model_def.get("vdn", False) or model_def.get("pdd", False))
            config = load_config()
            if not config["enabled"] or not _model_allowed["value"] or int(kwargs.get("guidance_phases", 1) or 1) <= 1:
                return original_get_loras(self, *args, **kwargs)
            if config["lora"] == NO_LORA:
                return [], []
            selected = {_basename(lora) for lora in kwargs.get("activated_loras") or ()}
            if config["lora"].lower() in selected:
                # Already in the user's list: keep their phase-1 multiplier, phase 2 is set in update_loras_slists.
                return [], []
            return [config["lora"]], [f"0;{config['lora_strength']:g}"]

        def generate(self, *args, **kwargs):
            config = load_config()
            active = (config["enabled"] and _model_allowed["value"] and int(kwargs.get("guide_phases", 1) or 1) > 1
                      and not getattr(self, "audio_only", False) and getattr(self.transformer, "pdd_num_steps", None) is None)
            if not active:
                return original_generate(self, *args, **kwargs)
            sigmas = build_sigmas(config["start_noise"], config["steps"], config["shift"])
            with _lock:
                previous_sigmas, previous_scale = pipeline.H3_PHASE_2_SIGMAS, pipeline.H3_TWO_PHASE_SCALE
                pipeline.H3_PHASE_2_SIGMAS = sigmas
                pipeline.H3_TWO_PHASE_SCALE = config["draft_scale"]
                kwargs["switch_threshold"] = sigmas[0]
                _run.update(active=True, lora=None if config["lora"] == NO_LORA else config["lora"].lower(),
                            strength=config["lora_strength"], loras_selected=kwargs.get("loras_selected"))
                print(f"[H3 Phase 2 Plus] {describe(config)}")
                try:
                    return original_generate(self, *args, **kwargs)
                finally:
                    pipeline.H3_PHASE_2_SIGMAS, pipeline.H3_TWO_PHASE_SCALE = previous_sigmas, previous_scale
                    _run.update(active=False, lora=None, loras_selected=None)

        def update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step=None, phase_switch_step2=None):
            # phase_switch_step == 0 is WanGP's phase-2 activation (after its own LoRA policy ran).
            if _run["active"] and phase_switch_step == 0 and _run["loras_selected"]:
                for index, lora in enumerate(_run["loras_selected"]):
                    name = _basename(lora)
                    if name == _run["lora"]:
                        slists_dict["phase2"][index] = float(_run["strength"])
                    elif "turbo" in name:
                        slists_dict["phase2"][index] = 0.0
                    else:
                        continue
                    slists_dict["shared"][index] = False
            return original_update_slists(trans, slists_dict, num_inference_steps, phase_switch_step=phase_switch_step, phase_switch_step2=phase_switch_step2)

        def upscale(self, latent, scale, target_size=None, abort_callback=None, progress_callback=None):
            args = (latent, scale, target_size, abort_callback, progress_callback)
            if not load_config()["single_pass_upscale"] or latent.shape[2] <= self.temporal_chunk_size:
                return original_upscale(self, *args)
            chunk_size = self.temporal_chunk_size
            try:
                self.temporal_chunk_size = latent.shape[2]
                return original_upscale(self, *args)
            except out_of_memory:
                pass
            finally:
                self.temporal_chunk_size = chunk_size
            # Only reached after an out-of-memory; retrying outside the except block releases the failed attempt's tensors first.
            torch.cuda.empty_cache()
            print(f"[H3 Phase 2 Plus] Not enough VRAM to upscale {latent.shape[2]} latent frames in one pass; "
                  f"using cross-faded {chunk_size}-frame segments instead.")
            return crossfaded_upscale(self, *args)

        pipeline.MiniMaxH3Pipeline.get_loras_transformer = get_loras_transformer
        pipeline.MiniMaxH3Pipeline.generate = generate
        pipeline.update_loras_slists = update_loras_slists
        MiniMaxH3LatentUpscaler.forward = upscale
        _patched = True
        print(f"[H3 Phase 2 Plus] Active. {describe(load_config())}")
