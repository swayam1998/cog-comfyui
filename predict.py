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
        
        # Update RandomNoise seed in node 25
        noise = workflow["25"]["inputs"]
        noise["noise_seed"] = kwargs["seed"]
        
        # Update FluxGuidance in node 58
        flux_guidance = workflow["58"]["inputs"]
        flux_guidance["guidance"] = kwargs["guidance"]
        
        # Update sampler in node 16
        sampler = workflow["16"]["inputs"]
        sampler["sampler_name"] = kwargs["sampler"]
        
        # Update scheduler in node 17
        scheduler = workflow["17"]["inputs"]
        scheduler["steps"] = kwargs["steps"]
        scheduler["scheduler"] = kwargs["scheduler"]

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the image to generate",
            default="EC$ style, a colorful digital illustration",
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
        guidance: float = Input(
            description="Flux Guidance scale",
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
            default="euler",
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
            guidance=guidance,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            steps=steps,
            batch_size=batch_size
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )