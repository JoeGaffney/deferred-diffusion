import base64
import os
from typing import Optional


def _convert_image_to_base64(image_path: str) -> Optional[str]:
    """Internal function that handles the actual base64 conversion."""
    if not image_path:
        return None

    if os.path.exists(image_path) is False:
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_str = base64.b64encode(image_bytes)
            base64_str = base64_str.decode("utf-8")  # Convert to a string

            print(f"Base64: {base64_str[:100]}...")
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
