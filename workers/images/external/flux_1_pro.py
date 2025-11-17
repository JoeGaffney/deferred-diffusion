from typing import Literal

from PIL import Image

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def get_size(
    context: ImageContext,
) -> Literal["1:1", "16:9", "9:16"]:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "16:9"
    elif dimension_type == "portrait":
        return "9:16"
    return "1:1"


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
        "input_image": convert_pil_to_bytes(context.color_image),
        "output_format": "png",
        "safety_tolerance": 6,
        "seed": context.data.seed,
    }

    output = replicate_run("black-forest-labs/flux-kontext-pro", payload)

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
    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)

    return text_to_image_call(context)
