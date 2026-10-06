import os

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from .h3_patch import NO_LORA, apply_patches, build_sigmas, describe, load_config, save_config

PlugIn_Name = "H3 Phase 2+"
PlugIn_Id = "H3Phase2Plus"
H3_LORA_MODEL_TYPE = "minimax_h3_ref2va_pruned"


class H3Phase2PlusPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = ("Controls the second phase of MiniMax H3 'Two Phases' latent-upscale generation: "
                            "choose the phase-2 LoRA and strength, the number of steps and the start noise.")

    def setup_ui(self):
        apply_patches()
        self.request_global("get_lora_dir")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def _lora_dir(self):
        try:
            return self.get_lora_dir(H3_LORA_MODEL_TYPE)
        except Exception:
            return os.path.join("loras", "minimax_h3")

    def _lora_choices(self, current=None):
        folder = self._lora_dir()
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".safetensors")) if os.path.isdir(folder) else []
        if current and current != NO_LORA and current not in files:
            files.append(current)
        return [NO_LORA] + files

    def create_ui(self):
        config = load_config()

        def preview(enabled, lora, strength, steps, start_noise, shift):
            return describe({"enabled": enabled, "lora": lora or NO_LORA, "lora_strength": strength,
                             "steps": int(steps), "start_noise": start_noise, "shift": shift})

        def save(enabled, lora, strength, steps, start_noise, shift):
            new_config = {"enabled": bool(enabled), "lora": lora or NO_LORA, "lora_strength": float(strength),
                          "steps": int(steps), "start_noise": float(start_noise), "shift": float(shift)}
            if new_config["lora"] != NO_LORA and not os.path.isfile(os.path.join(self._lora_dir(), new_config["lora"])):
                gr.Warning(f"{new_config['lora']} is not in {self._lora_dir()}; generation will fail until it is.")
            save_config(new_config)
            gr.Info("H3 Phase 2+ settings saved. They apply to the next generation.")
            return describe(load_config())

        def reset():
            from .h3_patch import DEFAULTS
            return (DEFAULTS["enabled"], DEFAULTS["lora"], DEFAULTS["lora_strength"], DEFAULTS["steps"],
                    DEFAULTS["start_noise"], DEFAULTS["shift"])

        with gr.Column():
            gr.Markdown(
                "### MiniMax H3 — Phase 2 (latent upscale refine)\n"
                "Applies when an H3 model runs with *Advanced Mode → General → Phases → Two Phases*. "
                "While enabled, it **replaces** WanGP's phase 2 (3 steps, forced LightX2V v0.1 Turbo at 1.0) and "
                "the *Phase 2 Noise Level Start* slider is ignored. PDD and VDN models are left untouched.\n\n"
                "Defaults mirror the ComfyUI workflow *Minimax H3 Cinematic Version*: 4 steps, denoise 0.25 at shift 12 "
                "(start noise 0.80), `lightx2v_hybrid-4to8step-Turbo_r48` at 0.7. "
                "To keep your look LoRAs active in phase 2, give them a two-value multiplier in the main tab, e.g. `0.7;0.7`."
            )
            enabled = gr.Checkbox(label="Enable H3 Phase 2+", value=config["enabled"])
            with gr.Row():
                lora = gr.Dropdown(label="Phase 2 LoRA (from loras/minimax_h3)", choices=self._lora_choices(config["lora"]),
                                   value=config["lora"], scale=4)
                refresh = gr.Button("Refresh list", scale=1)
            strength = gr.Slider(label="Phase 2 LoRA strength", minimum=0.0, maximum=1.5, step=0.05, value=config["lora_strength"])
            steps = gr.Slider(label="Phase 2 steps", minimum=1, maximum=12, step=1, value=config["steps"])
            start_noise = gr.Slider(label="Phase 2 start noise (sigma). 0.80 = ComfyUI denoise 0.25 at shift 12; lower keeps more of phase 1",
                                    minimum=0.30, maximum=0.99, step=0.005, value=config["start_noise"])
            with gr.Accordion("Advanced", open=False):
                shift = gr.Number(label="Schedule shift (H3 default 12)", value=config["shift"], minimum=1.0)
            summary = gr.Markdown(describe(config))
            with gr.Row():
                save_btn = gr.Button("Save", variant="primary")
                reset_btn = gr.Button("Reset to defaults")

        inputs = [enabled, lora, strength, steps, start_noise, shift]
        for component in inputs:
            component.change(preview, inputs=inputs, outputs=[summary], show_progress="hidden")
        refresh.click(lambda current: gr.update(choices=self._lora_choices(current)), inputs=[lora], outputs=[lora])
        save_btn.click(save, inputs=inputs, outputs=[summary])
        reset_btn.click(reset, outputs=inputs)
