import replicate
from PIL import Image

from common.replicate_helpers import process_replicate_output
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def main(context: ImageContext) -> Image.Image:
    if context.color_image is None:
        raise ValueError("No color image provided")

    try:
        payload = {
            "prompt": context.data.prompt,
            "input_image": convert_pil_to_bytes(context.color_image),
            "output_format": "png",
            "safety_tolerance": 6,
            "seed": context.data.seed,
        }
    except Exception as e:
        raise RuntimeError(f"Error calling Replicate API: {e}")

    output = replicate.run(context.data.model_path, input=payload)
    return process_replicate_output(output)
