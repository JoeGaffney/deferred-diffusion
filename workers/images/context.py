import copy
import tempfile
from typing import Literal

import torch
from PIL import Image

from common.logger import logger
from images.adapters import Adapters
from images.control_nets import ControlNets
from images.schemas import ImageRequest, ModelConfig, PipelineConfig
from utils.utils import (
    get_tmp_dir,
    load_image_if_exists,
    resize_image,
    save_copy_with_timestamp,
)

IMAGE_MODEL_CONFIG = {
    "sd1.5": ModelConfig(model_family="sd1.5", model_path="stable-diffusion-v1-5/stable-diffusion-v1-5"),
    "sdxl": ModelConfig(model_family="sdxl", model_path="stabilityai/stable-diffusion-xl-base-1.0"),
    "sdxl-refiner": ModelConfig(model_family="sdxl", model_path="stabilityai/stable-diffusion-xl-refiner-1.0"),
    "RealVisXL": ModelConfig(model_family="sdxl", model_path="SG161222/RealVisXL_V4.0"),
    "Fluently-XL": ModelConfig(model_family="sdxl", model_path="fluently/Fluently-XL-v4"),
    "juggernaut-xl": ModelConfig(model_family="sdxl", model_path="RunDiffusion/Juggernaut-XL-v9"),
    "sd3": ModelConfig(model_family="sd3", model_path="stabilityai/stable-diffusion-3-medium-diffusers"),
    "sd3.5": ModelConfig(model_family="sd3", model_path="stabilityai/stable-diffusion-3.5-medium"),
    "flux-schnell": ModelConfig(model_family="flux", model_path="black-forest-labs/FLUX.1-schnell"),
    "flux-dev": ModelConfig(model_family="flux", model_path="black-forest-labs/FLUX.1-dev"),
    "depth-anything": ModelConfig(
        model_family="depth_anything", model_path="depth-anything/Depth-Anything-V2-Large-hf"
    ),
    "segment-anything": ModelConfig(model_family="segment_anything", model_path="sam2.1_hiera_base_plus"),
    "sd-x4-upscaler": ModelConfig(model_family="sd_upscaler", model_path="stabilityai/stable-diffusion-x4-upscaler"),
    "gpt-image-1": ModelConfig(model_family="openai", model_path="gpt-image-1"),
    "runway/gen4_image": ModelConfig(model_family="runway", model_path="gen4_image"),
    "HiDream": ModelConfig(model_family="hidream", model_path="HiDream-ai/HiDream-I1-Full"),
}


def get_model_config(key: str) -> ModelConfig:
    config = IMAGE_MODEL_CONFIG.get(key)
    if not config:
        raise ValueError(f"Model config for {key} not found")
    return config


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.data = data
        self.model = data.model
        self.model_config = get_model_config(data.model)
        self.torch_dtype = torch.float16  # just keep float 16 for now
        self.orig_height = copy.copy(data.max_height)
        self.orig_width = copy.copy(data.max_width)
        self.generator = torch.Generator(device="cpu").manual_seed(self.data.seed)
        self.target_precision = data.target_precision

        # Round down to nearest multiple of 16
        self.division = 16
        self.width = (copy.copy(data.max_width) // self.division) * self.division
        self.height = (copy.copy(data.max_height) // self.division) * self.division

        self.color_image = load_image_if_exists(data.image)
        if self.color_image:
            self.color_image = resize_image(
                self.color_image, self.division, 1.0, self.data.max_width, self.data.max_height
            )
            self.orig_width, self.orig_height = self.color_image.size

            # NOTE base width and height now become the color image size masks and contolnet images are resized to this
            self.width, self.height = self.color_image.size

        # add our input mask image
        self.mask_image = load_image_if_exists(data.mask)
        if self.mask_image:
            self.mask_image = self.mask_image.convert("L")
            self.mask_image = self.mask_image.resize([self.width, self.height])

        # Initialize control nets and adapters
        self.control_nets = ControlNets(data.controlnets, self.model_config, self.width, self.height, self.torch_dtype)
        self.adapters = Adapters(data.ip_adapters, self.model_config, self.width, self.height)

    def get_pipeline_config(self) -> PipelineConfig:
        adapter_config = self.adapters.get_pipeline_config()

        return PipelineConfig(
            model_id=self.model_config.model_path,
            model_family=self.model_config.model_family,
            torch_dtype=self.torch_dtype,
            target_precision=self.target_precision,  # type: ignore
            use_safetensors=True,
            ip_adapter_models=adapter_config.get("models", ()),
            ip_adapter_subfolders=adapter_config.get("subfolders", ()),
            ip_adapter_weights=adapter_config.get("weights", ()),
            ip_adapter_image_encoder_model=adapter_config.get("image_encoder_model", ""),
            ip_adapter_image_encoder_subfolder=adapter_config.get("image_encoder_subfolder", ""),
        )

    def get_quality(self) -> Literal["low", "medium", "high"]:
        """Get quality setting based on number of inference steps."""
        if self.data.num_inference_steps < 20:
            return "low"
        elif self.data.num_inference_steps > 40:
            return "high"
        return "medium"

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def resize_image_to_orig(self, image: Image.Image, scale=1) -> Image.Image:
        return image.resize((self.orig_width * scale, self.orig_height * scale))

    def cleanup(self):
        self.control_nets.cleanup()

    # NOTE could be moved to utils
    def save_image(self, image):
        # Create a temporary file with .png extension
        with tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".png", delete=False) as tmp_file:
            # tmp_file will be closed automatically when exiting the with block
            image.save(tmp_file, format="PNG")
            path = tmp_file.name

        # Save the image to the temporary file
        logger.info(f"Image saved to temporary file: {path} size: {image.size}")
        save_copy_with_timestamp(path)

        return path
