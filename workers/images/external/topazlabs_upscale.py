from pathlib import Path
from typing import List

from PIL import Image

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> List[Path]:
    if context.color_image is None:
        raise ValueError("No color image provided")

    payload = {
        "image": convert_pil_to_bytes(context.color_image),
        "upscale_factor": "4x",
        "output_format": "png",
        "face_enhancement": False,
    }

    output = replicate_run("topazlabs/image-upscale", payload)

    processed_image = process_replicate_image_output(output)
    return [context.save_output(processed_image, index=0)]
