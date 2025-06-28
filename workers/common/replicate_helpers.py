from typing import Any

from diffusers.utils import load_image
from PIL import Image
from replicate.helpers import FileOutput


def process_replicate_output(output: Any) -> Image.Image:
    if not isinstance(output, FileOutput):
        raise ValueError(f"Expected output from replicate to be FileOutput, got {type(output)} {str(output)}")

    try:
        processed_image = load_image(output.url)
    except Exception as e:
        raise ValueError(f"Failed to process image from replicate: {str(e)} {output}")

    return processed_image
