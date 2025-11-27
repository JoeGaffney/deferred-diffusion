from typing import Any

import replicate
from diffusers.utils import load_image
from PIL import Image
from replicate.helpers import FileOutput


def replicate_run(model_path: str, payload: dict[str, Any]) -> Any:
    try:
        output = replicate.run(model_path, input=payload)
    except Exception as e:
        raise RuntimeError(f"Error calling Replicate API {model_path}: {e}")

    return output


def process_replicate_image_output(output: Any) -> Image.Image:
    if not isinstance(output, FileOutput):
        raise ValueError(f"Expected output from replicate to be FileOutput, got {type(output)} {str(output)}")

    try:
        processed_image = load_image(output.url)
    except Exception as e:
        raise ValueError(f"Failed to process image from replicate: {str(e)} {output}")

    return processed_image


def process_replicate_video_output(output: Any) -> str:
    """Process replicate video output and return the URL for download."""
    if not isinstance(output, FileOutput):
        raise ValueError(f"Expected output from replicate to be FileOutput, got {type(output)} {str(output)}")

    return output.url
