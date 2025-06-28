import replicate
from PIL import Image

from common.replicate_helpers import process_replicate_output
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> Image.Image:
    if context.color_image is None:
        raise ValueError("No color image provided")

    payload = {
        "prompt": context.data.prompt,
        "input_image": convert_pil_to_bytes(context.color_image),
        "output_format": "png",
        "safety_tolerance": 6,
        "seed": context.data.seed,
    }

    output = replicate.run("black-forest-labs/flux-kontext-pro", input=payload)
    return process_replicate_output(output)
