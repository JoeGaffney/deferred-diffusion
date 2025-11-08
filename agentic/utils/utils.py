import asyncio
import base64
import io
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from common.logger import logger

CACHE_DIR = Path(tempfile.gettempdir()) / "deferred_diffusion_cache"
CACHE_DIR.mkdir(exist_ok=True)


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


def load_image_from_base64(base64_bytes: str) -> Image.Image:

    try:
        # Convert bytes to a PIL image
        tmp_bytes = base64.b64decode(base64_bytes)
        imageFile = Image.open(io.BytesIO(tmp_bytes))
        image = imageFile.convert("RGB")  # Ensure the image is in RGB mode
        logger.info(f"Image loaded from Base64 bytes, size: {image.size}")
        return image
    except Exception as e:
        raise ValueError(f"Invalid Base64 data: {type(base64_bytes)} {e}") from e


# async def poll_until_resolved(id: str, timeout=300, poll_interval=3) -> AsyncResult:
#     """
#     Poll a Celery task until it's complete, max attempts reached, or timeout.

#     Returns:
#         AsyncResult object with task result/status
#     """
#     # Ensure id is a string
#     if not isinstance(id, str):
#         logger.warning(f"id is not a string, converting from {type(id)}")
#         id = str(id)

#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         logger.info(f"Polling task {id}... {timeout - (time.time() - start_time)} seconds left")
#         # Get task result
#         result = AsyncResult(id, app=celery_app)

#         if result.ready():
#             if result.failed():
#                 error_msg = str(result.result)
#                 logger.error(f"Task {id} failed: {error_msg}")
#                 return result
#             elif result.successful():
#                 return result

#         # Wait before polling again
#         await asyncio.sleep(poll_interval)

#     logger.warning(f"Task {id} polling timeout ({timeout} seconds) reached")

#     # Get final status before returning
#     result = AsyncResult(id, app=celery_app)
#     return result
