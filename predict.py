# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Update positive prompt in node 6
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        # Update negative prompt in node 53
        negative_prompt = workflow["53"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"
        
        # Update Empty Latent Image parameters in node 5
        latent_image = workflow["5"]["inputs"]
        latent_image["batch_size"] = kwargs["batch_size"]
        latent_image["width"] = kwargs["width"]
        latent_image["height"] = kwargs["height"]
        
        # Update KSampler parameters in node 55
        ksampler = workflow["55"]["inputs"]
        ksampler["seed"] = kwargs["seed"]
        ksampler["steps"] = kwargs["steps"]
        ksampler["cfg"] = kwargs["cfg"]
        ksampler["sampler_name"] = kwargs["sampler"]
        ksampler["scheduler"] = kwargs["scheduler"]
        ksampler["denoise"] = kwargs["denoise"]
        
        # Update FluxGuidance in node 59
        flux_guidance = workflow["59"]["inputs"]
        flux_guidance["guidance"] = kwargs["guidance"]
        
        # Update LoRA strength if requested
        if "lora_strength" in kwargs and "54" in workflow:
            lora_loader = workflow["54"]["inputs"]
            lora_loader["strength_model"] = kwargs["lora_strength"]
            lora_loader["strength_clip"] = kwargs["lora_strength"]

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the image to generate",
            default="ohwx woman with short blonde hair is a food influencer. She is cooking a healthy meal in her bright brooklyn apartment.",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        width: int = Input(
            description="Width of the generated image",
            default=896,
            ge=384,
            le=1536,
        ),
        height: int = Input(
            description="Height of the generated image",
            default=1152,
            ge=384,
            le=1536,
        ),
        cfg: float = Input(
            description="Classifier-Free Guidance scale in KSampler",
            default=1.0,
            ge=1.0,
            le=20.0,
        ),
        guidance: float = Input(
            description="Flux Guidance scale - controls how closely the image follows the prompt",
            default=3.5,
            ge=1.0,
            le=10.0,
        ),
        sampler: str = Input(
            description="Which sampler to use for the diffusion process",
            choices=["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
                    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", 
                    "ddim", "uni_pc", "uni_pc_bh2"],
            default="uni_pc",
        ),
        scheduler: str = Input(
            description="Scheduler to use for the diffusion process",
            choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],
            default="normal",
        ),
        steps: int = Input(
            description="Number of diffusion steps to run",
            default=30,
            ge=10,
            le=100,
        ),
        denoise: float = Input(
            description="Denoising strength - lower values preserve more of the initial noise",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA effect (applies to both model and CLIP)",
            default=1.0,
            ge=0.0,
            le=2.0,
        ),
        batch_size: int = Input(
            description="Number of images to generate in a batch",
            default=1,
            ge=1,
            le=8,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            cfg=cfg,
            guidance=guidance,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            steps=steps,
            denoise=denoise,
            lora_strength=lora_strength,
            batch_size=batch_size
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )