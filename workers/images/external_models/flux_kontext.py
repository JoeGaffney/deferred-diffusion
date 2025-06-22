import replicate
from PIL import Image

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

    output = replicate.run("black-forest-labs/flux-kontext-pro", input=input)
    if output is None:
        raise ValueError("No image data returned from replicate API")

    print(f"Output from Flux Kontext: {output}")
    processed_image = Image.open(output.read())
    return processed_image
