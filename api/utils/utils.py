import asyncio
import base64
import io
import time

from celery.result import AsyncResult
from PIL import Image

from common.logger import logger
from worker import celery_app


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


async def poll_until_resolved(id: str, timeout=300, poll_interval=3) -> AsyncResult:
    """
    Poll a Celery task until it's complete, max attempts reached, or timeout.

    Returns:
        AsyncResult object with task result/status
    """
    # Ensure id is a string
    if not isinstance(id, str):
        logger.warning(f"id is not a string, converting from {type(id)}")
        id = str(id)

    start_time = time.time()
    while time.time() - start_time < timeout:
        logger.info(f"Polling task {id}... {timeout - (time.time() - start_time)} seconds left")
        # Get task result
        result = AsyncResult(id, app=celery_app)

        if result.ready():
            if result.failed():
                error_msg = str(result.result)
                logger.error(f"Task {id} failed: {error_msg}")
                return result
            elif result.successful():
                return result

        # Wait before polling again
        await asyncio.sleep(poll_interval)

    logger.warning(f"Task {id} polling timeout ({timeout} seconds) reached")

    # Get final status before returning
    result = AsyncResult(id, app=celery_app)
    return result
