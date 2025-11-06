import copy
import tempfile
from typing import Literal, Tuple

import torch
from pydantic import BaseModel

from common.logger import logger
from images.adapters import Adapters
from images.control_nets import ControlNets
from images.schemas import ImageRequest, ModelName
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    load_image_if_exists,
    resize_image,
    save_copy_with_timestamp,
)


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.model = data.model
        self.data = data
        self.generator = torch.Generator(device="cpu").manual_seed(self.data.seed)
        self.width = copy.copy(data.width)
        self.height = copy.copy(data.height)
        self.color_image = load_image_if_exists(data.image)
        if self.color_image:
            self.width, self.height = self.color_image.size

        # add our input mask image
        self.mask_image = load_image_if_exists(data.mask)
        if self.mask_image:
            self.mask_image = self.mask_image.convert("L")
            self.mask_image = self.mask_image.resize([self.width, self.height])

        # Initialize control nets and adapters
        self.control_nets = ControlNets(data.references, self.model, self.width, self.height)
        self.adapters = Adapters(data.references, self.model, self.width, self.height)

    def ensure_divisible(self, value: int):
        # Adjust width and height to be divisible by the specified value
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)
        if self.mask_image:
            self.mask_image = self.mask_image.resize([self.width, self.height])
        if self.color_image:
            self.color_image = self.color_image.resize([self.width, self.height])

    def ensure_max_dimension(self, delta: int = 1024):
        if self.width < delta or self.height < delta:
            # Calculate scale factor to get closest dimension to 1024
            scale_factor = max(delta / self.width, delta / self.height)

            # Apply scaling and ensure divisibility by 16
            self.width = int(self.width * scale_factor)
            self.height = int(self.height * scale_factor)

    def get_generation_mode(self) -> Literal["text_to_image", "img_to_img", "img_to_img_inpainting"]:
        """Get generation mode, can be overridden by specific models."""
        if self.data.image is None:
            return "text_to_image"
        elif self.data.mask:
            return "img_to_img_inpainting"
        else:
            return "img_to_img"

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def cleanup(self):
        self.control_nets.cleanup()

    def get_reference_images(self) -> list:
        result = []
        for references in self.data.references:
            image = load_image_if_exists(references.image)
            if image:
                result.append(image)

        return result

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
