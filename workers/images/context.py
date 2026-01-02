import copy
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from common.config import settings
from common.logger import get_task_id, logger, task_log
from images.schemas import ImageRequest
from utils.utils import ensure_divisible, image_crop, image_resize, load_image_if_exists


class ImageContext:
    def __init__(self, data: ImageRequest, task_id: str = get_task_id()):
        self.model = data.model
        self.task_id = task_id
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

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def get_reference_images(self) -> list:
        result = []
        for references in self.data.references:
            image = load_image_if_exists(references.image)
            if image:
                result.append(image)

        return result

    def save_image(self, image):
        # Create a temporary file with .png extension
        with tempfile.NamedTemporaryFile(dir=settings.storage_dir, suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file, format="PNG")
            path = tmp_file.name
            logger.info(f"Image saved at {path}")

        return path

    def save_output(self, image: Image.Image, index: int = 0) -> Path:
        # deterministic relative path
        rel_path = Path(self.model) / f"{self.task_id}-{index}.png"
        abs_path = settings.storage_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image.save(abs_path, format="PNG")
            logger.info(f"Image saved at {abs_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save image at {abs_path}: {e}")

        return abs_path  # return abs_path for now
