from PIL import Image

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> Image.Image:
    payload = {
        "prompt": context.data.prompt,
        "size": "2K",
        "aspect_ratio": "match_input_image",
        "sequential_image_generation": "disabled",
        "max_images": 1,
    }

    reference_images = []
    if context.color_image:
        reference_images.append(convert_pil_to_bytes(context.color_image))

    for current in context.get_reference_images():
        reference_images.append(convert_pil_to_bytes(current))

    # Add input images if available
    if reference_images != []:
        payload["image_input"] = reference_images

    output = replicate_run("bytedance/seedream-4", payload)

    # as this can return a list, we just take the first item
    if isinstance(output, list):
        output = output[0]

    return process_replicate_image_output(output)
