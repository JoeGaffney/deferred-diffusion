import copy
import tempfile
from typing import Literal

import torch

from common.logger import logger, task_log
from images.adapters import Adapters
from images.control_nets import ControlNets
from images.schemas import ImageRequest
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    image_crop,
    image_resize,
    load_image_if_exists,
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
            self.mask_image = image_resize(self.mask_image, (self.width, self.height))

        # Initialize control nets and adapters
        self.control_nets = ControlNets(data.references, self.model, self.width, self.height)
        self.adapters = Adapters(data.references, self.model, self.width, self.height)

        task_log(
            f"Context created {self.model}, {self.width}x{self.height}",
        )

    def ensure_divisible(self, value: int):
        # Adjust width and height to be divisible by the specified value
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)

        if self.mask_image:
            self.mask_image = image_crop(self.mask_image, (self.width, self.height))
        if self.color_image:
            self.color_image = image_crop(self.color_image, (self.width, self.height))

        # also adjust control net images
        self.control_nets.crop_all_images((self.width, self.height))

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

    def save_image(self, image):
        # Create a temporary file with .png extension
        with tempfile.NamedTemporaryFile(dir=get_tmp_dir(self.model), suffix=".png", delete=False) as tmp_file:
            # tmp_file will be closed automatically when exiting the with block
            image.save(tmp_file, format="PNG")
            path = tmp_file.name
            logger.info(f"Image saved at {path}")

        return path
