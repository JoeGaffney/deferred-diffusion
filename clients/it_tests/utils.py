import base64
import os
from typing import Optional


def base64_to_image(base64_str: str, output_path: str):
    """Convert a base64 string to an image and save it to the specified path."""

    try:
        # Handle both string and bytes input
        if isinstance(base64_str, str):
            if "," in base64_str and ";base64," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            base64_bytes = base64_str.encode("utf-8")
        else:
            base64_bytes = base64_str

        # Decode the base64 to binary
        image_bytes = base64.b64decode(base64_bytes)

        # Create directory if it doesn't exist and create_dir is True
        dir_path = os.path.dirname(output_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Write the bytes to the specified file path
        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)

    except Exception as e:
        raise ValueError(f"Error saving base64 to image {output_path}: {str(e)}") from e


def save_image_and_assert_file_exists(result, output_name):
    if os.path.exists(output_name):
        os.remove(output_name)

    base64_to_image(result, output_name)
    assert os.path.exists(output_name), f"Output file {output_name} was not created."


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
