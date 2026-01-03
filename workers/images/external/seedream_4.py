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
        "size": "2K",
        "aspect_ratio": get_size(context),
        "sequential_image_generation": "disabled",
        "max_images": 1,
        "enhance_prompt": False,
    }

    reference_images = []
    if context.color_image:
        reference_images.append(convert_pil_to_bytes(context.color_image))
        payload["aspect_ratio"] = "match_input_image"

    for current in context.get_reference_images():
        reference_images.append(convert_pil_to_bytes(current))

    # Add input images if available
    if reference_images != []:
        payload["image_input"] = reference_images

    output = replicate_run("bytedance/seedream-4.5", payload)

    # as this can return a list, we just take the first item
    if isinstance(output, list):
        output = output[0]

    processed_image = process_replicate_image_output(output)
    return [context.save_output(processed_image, index=0)]
