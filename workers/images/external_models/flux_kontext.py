from typing import Any

import replicate
from diffusers.utils import load_image
from PIL import Image
from replicate.helpers import FileOutput

from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> Image.Image:
    if context.color_image is None:
        raise ValueError("No color image provided")

    input = {
        "prompt": context.data.prompt,
        "input_image": convert_pil_to_bytes(context.color_image),
        "output_format": "png",
        "safety_tolerance": 6,
        "seed": context.data.seed,
    }

    output: FileOutput | Any = replicate.run("black-forest-labs/flux-kontext-pro", input=input)
    if not isinstance(output, FileOutput):
        raise ValueError(f"Expected output from replicate to be FileOutput, got {type(output)} {str(output)}")

    try:
        processed_image = load_image(output.url)
    except Exception as e:
        raise ValueError(f"Failed to process image from replicate: {str(e)} {output}")
    return processed_image
