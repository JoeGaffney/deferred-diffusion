from typing import Literal

from PIL import Image

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def get_size(
    context: ImageContext,
) -> Literal["1:1", "16:9", "9:16"]:
    size = "1:1"
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        size = "16:9"
    elif dimension_type == "portrait":
        size = "9:16"
    return size


def text_to_image_call(context: ImageContext) -> Image.Image:

    payload = {
        "prompt": context.data.prompt,
        "aspect_ratio": get_size(context),
        "output_format": "png",
        "safety_tolerance": 6,
        "seed": context.data.seed,
        "raw": True,
    }

    output = replicate_run("black-forest-labs/flux-1.1-pro", payload)

    return process_replicate_image_output(output)


def image_to_image_call(context: ImageContext) -> Image.Image:
    if context.color_image is None:
        raise ValueError("No color image provided")

    payload = {
        "prompt": context.data.prompt,
        "image_prompt": convert_pil_to_bytes(context.color_image),
        "output_format": "png",
        "image_prompt_strength": context.data.strength,
        "safety_tolerance": 6,
        "seed": context.data.seed,
        "aspect_ratio": get_size(context),
        "raw": True,
    }

    output = replicate_run("black-forest-labs/flux-1.1-pro", payload)

    return process_replicate_image_output(output)


def inpainting_call(context: ImageContext) -> Image.Image:
    if context.color_image is None or context.mask_image is None:
        raise ValueError("No color image or mask image provided")

    payload = {
        "prompt": context.data.prompt,
        "image": convert_pil_to_bytes(context.color_image),
        "mask": convert_pil_to_bytes(context.mask_image),
        "output_format": "png",
        "safety_tolerance": 6,
        "seed": context.data.seed,
    }

    output = replicate_run("black-forest-labs/flux-fill-pro", payload)

    return process_replicate_image_output(output)


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    # OpenAI-specific handling of text_to_image with adapters
    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Invalid mode {mode} for OpenAI API")
