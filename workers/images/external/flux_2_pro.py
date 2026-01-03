from pathlib import Path
from typing import List, Literal

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


def main(context: ImageContext) -> List[Path]:
    payload = {
        "prompt": context.data.cleaned_prompt,
        "resolution": "2 MP",
        "aspect_ratio": get_size(context),
        "safety_tolerance": 5.0,
        "output_format": "png",
        "seed": context.data.seed,
    }

    reference_images = []
    if context.mask_image and context.color_image:
        payload["prompt"] = (
            "Use image 1 for the mask region for inpainting. And use image 2 for the base image only alter the mask region and aim for a seamless blend. "
            + context.data.cleaned_prompt
        )
        reference_images.append(convert_pil_to_bytes(context.mask_image))
        reference_images.append(convert_pil_to_bytes(context.color_image))
    else:
        if context.color_image:
            reference_images.append(convert_pil_to_bytes(context.color_image))
            payload["aspect_ratio"] = "match_input_image"

        for current in context.get_reference_images():
            reference_images.append(convert_pil_to_bytes(current))

    # Add input images if available
    if reference_images != []:
        payload["input_images"] = reference_images

    output = replicate_run("black-forest-labs/flux-2-pro", payload)
    processed_image = process_replicate_image_output(output)
    return [context.save_output(processed_image, index=0)]
