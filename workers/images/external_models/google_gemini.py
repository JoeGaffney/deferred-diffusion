from typing import Literal

from PIL import Image

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes, pill_to_base64


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
    payload = {"prompt": context.data.prompt, "output_format": "png", "aspect_ratio": get_size(context)}

    output = replicate_run("google/gemini-2.5-flash-image", payload)

    return process_replicate_image_output(output)


def image_to_image_call(context: ImageContext) -> Image.Image:
    # gather all possible reference images we piggy back on the ipdapter images
    reference_images = []
    if context.color_image:
        reference_images.append(convert_pil_to_bytes(context.color_image))

    for current in context.get_reference_images():
        reference_images.append(convert_pil_to_bytes(current))

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    payload = {
        "prompt": context.data.prompt,
        "image_input": reference_images,
        "output_format": "png",
    }
    output = replicate_run("google/nano-banana", payload)

    return process_replicate_image_output(output)


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        if context.get_reference_images() != []:
            return image_to_image_call(context)
        return text_to_image_call(context)
    elif mode == "img_to_img" or mode == "inpainting":
        return image_to_image_call(context)

    raise ValueError(f"Invalid mode {mode} for RunwayML API")
