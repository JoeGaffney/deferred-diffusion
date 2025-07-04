import copy
import tempfile
from typing import Literal, Tuple

import torch
from pydantic import BaseModel

from common.logger import logger
from images.adapters import Adapters
from images.control_nets import ControlNets
from images.schemas import ImageRequest, ModelFamily
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    load_image_if_exists,
    resize_image,
    save_copy_with_timestamp,
)


class PipelineConfig(BaseModel):
    model_id: str
    model_family: ModelFamily
    ip_adapter_models: Tuple[str, ...]
    ip_adapter_subfolders: Tuple[str, ...]
    ip_adapter_weights: Tuple[str, ...]
    ip_adapter_image_encoder_model: str
    ip_adapter_image_encoder_subfolder: str

    class Config:
        frozen = True  # Makes the model immutable/hashable
        arbitrary_types_allowed = True  # Needed for torch.dtype

    def __hash__(self):
        return hash(
            (
                self.model_id,
                self.ip_adapter_models,
                self.ip_adapter_subfolders,
                self.ip_adapter_weights,
                self.ip_adapter_image_encoder_model,
                self.ip_adapter_image_encoder_subfolder,
            )
        )


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.data = data
        self.model = data.model
        self.orig_height = copy.copy(data.height)
        self.orig_width = copy.copy(data.width)
        self.generator = torch.Generator(device="cpu").manual_seed(self.data.seed)

        # Round down to nearest multiple of the eg. 8, 16, 32, etc.
        self.division = data.model_divisor
        self.width = ensure_divisible(copy.copy(data.width), self.division)
        self.height = ensure_divisible(copy.copy(data.height), self.division)
        self.color_image = load_image_if_exists(data.image)
        if self.color_image:
            self.color_image = resize_image(self.color_image, self.division, 1.0, 2048, 2048)
            self.orig_width, self.orig_height = self.color_image.size

            # NOTE base width and height now become the color image size masks and contolnet images are resized to this
            self.width, self.height = self.color_image.size

        # add our input mask image
        self.mask_image = load_image_if_exists(data.mask)
        if self.mask_image:
            self.mask_image = self.mask_image.convert("L")
            self.mask_image = self.mask_image.resize([self.width, self.height])

        # Initialize control nets and adapters
        self.control_nets = ControlNets(data.controlnets, self.data.model_family, self.width, self.height)
        self.adapters = Adapters(data.ip_adapters, self.data.model_family, self.width, self.height)

    def get_generation_mode(self) -> Literal["text_to_image", "img_to_img", "img_to_img_inpainting"]:
        """Get generation mode, can be overridden by specific models."""
        if self.data.image is None:
            return "text_to_image"
        elif self.data.mask:
            return "img_to_img_inpainting"
        else:
            return "img_to_img"

    def get_pipeline_config(self) -> PipelineConfig:
        adapter_config = self.adapters.get_pipeline_config()

        # check if we should use inpainting model
        generation_mode = self.get_generation_mode()
        model_id = self.data.model_path
        if generation_mode == "img_to_img_inpainting":
            model_id = self.data.model_path_inpainting

        return PipelineConfig(
            model_id=model_id,
            model_family=self.data.model_family,
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
