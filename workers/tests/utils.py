import base64
import json
import os
from pathlib import Path
from typing import List, Optional


def asset_outputs_exists(outputs: List[Path]):
    """Check if all output asset files exist."""
    assert len(outputs) > 0, "No output files to check."
    for output in outputs:
        assert output.exists(), "Output file path does not exist."
        assert os.path.getsize(output) > 100, f"Output file {output} is empty."


def _convert_image_to_base64(image_path: str) -> Optional[str]:
    """Internal function that handles the actual base64 conversion."""
    if not image_path:
        return None

    if os.path.exists(image_path) is False:
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_bytes = base64.b64encode(image_bytes)
            base64_str = base64_bytes.decode("utf-8")  # Convert to a string

            print(f"Base64: {base64_str[:100]}... length: {len(base64_str)}")
            return base64_str
    except Exception as e:
        raise ValueError(f"Error encoding image {image_path}: {str(e)}")


def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64. Raises if conversion fails."""
    result = _convert_image_to_base64(image_path)
    if result is None:
        raise ValueError(f"Failed to encode required image to base64: {image_path}")
    return result


def optional_image_to_base64(image_path: str) -> Optional[str]:
    """Convert an image file to base64. Returns None if conversion fails."""
    return _convert_image_to_base64(image_path)


def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""

    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data
