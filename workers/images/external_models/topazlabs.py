import replicate
from PIL import Image

from common.replicate_helpers import process_replicate_output
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> Image.Image:
    if context.color_image is None:
        raise ValueError("No color image provided")

    payload = {
        "image": convert_pil_to_bytes(context.color_image),
        "upscale_factor": "4x",
        "output_format": "png",
        "face_enhancement": False,
    }

    try:
        output = replicate.run(context.data.model_path, input=payload)
    except Exception as e:
        raise RuntimeError(f"Error calling Replicate API: {e}")

    return process_replicate_output(output)
