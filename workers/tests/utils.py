import base64
import json
import os
import tempfile
from typing import Optional

from PIL import Image

from common.config import settings
from utils.utils import ensure_path_exists


def setup_output_file(model_id, mode, suffix="", extension="png"):
    """Prepare output path and delete existing file if needed."""
    model_id_nice = model_id.replace("/", "_").replace(":", "_")
    suffix_nice = f"_{suffix}" if suffix != "" else ""

    subdir = os.path.join(settings.storage_dir, "tests")
    os.makedirs(subdir, exist_ok=True)

    output_name = f"{subdir}/{model_id_nice}/{model_id_nice}_{mode}{suffix_nice}.{extension}"

    if os.path.exists(output_name):
        os.remove(output_name)
    ensure_path_exists(output_name)
    return output_name


def assert_file_exists(output_name: str, size: int = 100) -> None:
    """Assert that a file exists at the given path."""
    assert os.path.exists(output_name), f"Output file {output_name} was not created."
    assert os.path.getsize(output_name) > size, f"Output file {output_name} is empty."


def save_image_and_assert_file_exists(result, output_name):
    """Save image result and verify it exists."""
    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    assert_file_exists(output_name)


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
