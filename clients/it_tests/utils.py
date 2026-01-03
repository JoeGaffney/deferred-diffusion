import base64
import os
from typing import List, Optional

import httpx


def get_output_dir() -> str:
    output_dir = "../tmp/it-tests/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def asset_outputs_exists(outputs: List[str]):
    """Check if all output asset files exist."""
    assert len(outputs) > 0, "No output files to check."

    for i, url in enumerate(outputs):
        print(f"Checking output file from URL: {url}")
        # Download the file
        try:
            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.content
        except Exception as e:
            raise AssertionError(f"Failed to download {url}: {e}")

        assert len(content) > 100, f"Output file from {url} is too small ({len(content)} bytes)."

        # Save it to output_dir
        # Try to get a filename from the URL or use a default
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = f"output_{i}.bin"

        output_path = os.path.join(get_output_dir(), filename)
        with open(output_path, "wb") as f:
            f.write(content)

        assert os.path.exists(output_path), f"Output file {output_path} was not saved."
        assert os.path.getsize(output_path) > 100, f"Output file {output_path} is empty."


def assert_logs_exist(logs):
    print(f"Logs from generation: {logs}")
    assert logs is not None
    assert isinstance(logs, list)
    assert len(logs) > 0


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


image_a = image_to_base64("../../assets/color_v001.jpeg")
image_b = image_to_base64("../../assets/style_v001.jpeg")
image_c = image_to_base64("../../assets/color_v002.png")
video_a = image_to_base64("../../assets/video_v001.mp4")
