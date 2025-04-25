import copy
from typing import List, Literal

import torch

from common.exceptions import ControlNetConfigError, IPAdapterConfigError
from common.logger import logger
from image.control_net import ControlNet
from image.ip_adapter import IpAdapter
from image.schemas import ImageRequest, ModelConfig, PipelineConfig
from utils.utils import (
    ensure_path_exists,
    load_image_if_exists,
    resize_image,
    save_copy_with_timestamp,
)

IMAGE_MODEL_CONFIG = {
    "sd1.5": {"family": "sd1.5", "model_path": "stable-diffusion-v1-5/stable-diffusion-v1-5", "mode": "auto"},
    "sdxl": {"family": "sdxl", "model_path": "stabilityai/stable-diffusion-xl-base-1.0", "mode": "auto"},
    "sdxl-refiner": {"family": "sdxl", "model_path": "stabilityai/stable-diffusion-xl-refiner-1.0", "mode": "auto"},
    "RealVisXL": {"family": "sdxl", "model_path": "SG161222/RealVisXL_V4.0", "mode": "auto"},
    "Fluently-XL": {"family": "sdxl", "model_path": "fluently/Fluently-XL-v4", "mode": "auto"},
    "sd3": {"family": "sd3", "model_path": "stabilityai/stable-diffusion-3-medium-diffusers", "mode": "auto"},
    "sd3.5": {"family": "sd3", "model_path": "stabilityai/stable-diffusion-3.5-medium", "mode": "auto"},
    "flux-schnell": {
        "family": "flux",
        "model_path": "black-forest-labs/FLUX.1-schnell",
        "transformer_guf_path": "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q5_0.gguf",
        "mode": "auto",
    },
    "depth-anything": {
        "family": "depth_anything",
        "model_path": "depth-anything/Depth-Anything-V2-Large-hf",
        "mode": "depth",
    },
    "segment-anything": {"family": "segment_anything", "model_path": "sam2.1_hiera_base_plus", "mode": "mask"},
    "sd-x4-upscaler": {
        "family": "stable-diffusion-x4-upscaler",
        "model_path": "stabilityai/stable-diffusion-x4-upscaler",
        "mode": "upscale",
    },
    "gpt-image-1": {
        "family": "openai",
        "model_path": "gpt-image-1",
        "mode": "auto",
    },
}


def get_model_config(key: str) -> ModelConfig:
    config = IMAGE_MODEL_CONFIG.get(key)
    if not config:
        raise ValueError(f"Model config for {key} not found")

    return ModelConfig(
        model_path=config.get("model_path", ""),
        model_family=config.get("family", ""),
        transformer_guf_path=config.get("transformer_guf_path", ""),
        mode=config["mode"],
    )


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.data = data
        self.model = data.model
        self.model_config = get_model_config(data.model)
        self.torch_dtype = torch.float16  # just keep float 16 for now
        self.orig_height = copy.copy(data.max_height)
        self.orig_width = copy.copy(data.max_width)
        self.generator = torch.Generator(device="cpu").manual_seed(self.data.seed)
        self.optimize_low_vram = bool(data.optimize_low_vram)

        # Round down to nearest multiple of 16
        self.division = 16
        self.width = (copy.copy(data.max_width) // self.division) * self.division
        self.height = (copy.copy(data.max_height) // self.division) * self.division

        self.color_image = load_image_if_exists(data.input_image_path)
        if self.color_image:
            self.color_image = resize_image(
                self.color_image, self.division, 1.0, self.data.max_width, self.data.max_height
            )
            self.orig_width, self.orig_height = self.color_image.size

            # NOTE base width and height now become the color image size masks and contolnet images are resized to this
            self.width, self.height = self.color_image.size

        # add our input mask image
        self.mask_image = load_image_if_exists(data.input_mask_path)
        if self.mask_image:
            self.mask_image = self.mask_image.convert("L")
            self.mask_image = self.mask_image.resize([self.width, self.height])

        # add our control nets
        self.controlnets: List[ControlNet] = []
        for current in data.controlnets:
            try:
                tmp = ControlNet(current, self.model_config, self.width, self.height, torch_dtype=self.torch_dtype)
                self.controlnets.append(tmp)
            except ControlNetConfigError as e:
                logger.error(e)
                continue

        self.controlnets_enabled = len(self.controlnets) > 0

        # add our ip adapters
        self.ip_adapters: List[IpAdapter] = []
        for current in data.ip_adapters:
            try:
                tmp = IpAdapter(current, self.model_config, self.width, self.height)
                self.ip_adapters.append(tmp)
            except IPAdapterConfigError as e:
                logger.error(e)
                continue

        self.ip_adapters_enabled = len(self.ip_adapters) > 0

    def get_pipeline_config(self) -> PipelineConfig:
        # it wants in a array for multiple adapters
        models = []
        subfolders = []
        weights = []
        image_encoder_model = ""
        image_encoder_subfolder = ""

        if self.ip_adapters_enabled == True:
            # NOTE do we validate against clashes here? Or allow passing through the natural errors?
            for ip_adapter in self.ip_adapters:
                models.append(ip_adapter.config.model)
                subfolders.append(ip_adapter.config.subfolder)
                weights.append(ip_adapter.config.weight_name)
                if ip_adapter.config.image_encoder:
                    image_encoder_model = ip_adapter.config.model
                    image_encoder_subfolder = ip_adapter.config.image_encoder_subfolder

        return PipelineConfig(
            model_id=self.model_config.model_path,
            model_family=self.model_config.model_family,
            model_transformer_guf_path=self.model_config.transformer_guf_path,
            torch_dtype=self.torch_dtype,
            optimize_low_vram=self.optimize_low_vram,
            use_safetensors=True,
            ip_adapter_models=tuple(models),
            ip_adapter_subfolders=tuple(subfolders),
            ip_adapter_weights=tuple(weights),
            ip_adapter_image_encoder_model=image_encoder_model,
            ip_adapter_image_encoder_subfolder=image_encoder_subfolder,
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

    def save_image(self, image):
        ensure_path_exists(self.data.output_image_path)
        path = self.data.output_image_path
        image.save(path)
        logger.info(f"Image saved at {path} size: {image.size}")

        save_copy_with_timestamp(path)
        return path

    def resize_image_to_orig(self, image, scale=1):
        return image.resize((self.orig_width * scale, self.orig_height * scale))

    def get_controlnet_images(self):
        if self.controlnets_enabled == False:
            return []

        images = []
        for controlnet in self.controlnets:
            images.append(controlnet.image)
        return images

    def get_controlnet_conditioning_scales(self):
        if self.controlnets_enabled == False:
            return 0.8

        controlnet_conditioning_scales = []
        for controlnet in self.controlnets:
            controlnet_conditioning_scales.append(controlnet.conditioning_scale)
        return controlnet_conditioning_scales

    def get_loaded_controlnets(self):
        if self.controlnets_enabled == False:
            return []

        loaded_controlnets = []
        for controlnet in self.controlnets:
            loaded_controlnets.append(controlnet.get_loaded_controlnet())
        return loaded_controlnets

    def cleanup(self):
        if self.controlnets_enabled:
            for controlnet in self.controlnets:
                controlnet.cleanup()

    def get_ip_adapter_images(self):
        if self.ip_adapters_enabled == False:
            return []

        images = []
        for ip_adapter in self.ip_adapters:
            images.append(ip_adapter.image)
        return images

    def get_ip_adapter_masks(self):
        if self.ip_adapters_enabled == False:
            return []

        masks = []
        for ip_adapter in self.ip_adapters:
            masks.append(ip_adapter.get_mask())

        return masks

    def set_ip_adapter_scale(self, pipe):
        if self.ip_adapters_enabled == True:
            scales = []
            for ip_adapter in self.ip_adapters:
                scales.append(ip_adapter.get_scale_layers())
            pipe.set_ip_adapter_scale(scales)
        return pipe
