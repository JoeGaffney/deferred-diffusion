import asyncio
import base64
import io

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


async def poll_until_complete(id, max_attempts=90, polling_interval=10) -> AsyncResult:
    """
    Poll a Celery task until it's complete, max attempts reached, or timeout.

    Args:
        id: The ID of the Celery task to poll
        max_attempts: Maximum number of polling attempts (default: 30)
        polling_interval: How often to check task status in seconds (default: 1)

    Returns:
        AsyncResult object with task result/status
    """
    # Ensure id is a string
    if not isinstance(id, str):
        logger.warning(f"id is not a string, converting from {type(id)}")
        id = str(id)

    logger.debug(f"Polling task with ID: {id}, max attempts: {max_attempts}")

    # Use iteration count instead of time-based approach
    for attempt in range(max_attempts):
        # Get task result
        result = AsyncResult(id, app=celery_app)

        if result.ready():
            if result.failed():
                error_msg = str(result.result)
                logger.error(f"Task {id} failed: {error_msg}")
                return result
            elif result.successful():
                logger.info(f"Task {id} completed successfully after {attempt + 1} attempts")
                return result

        # Wait before polling again
        await asyncio.sleep(polling_interval)

    # If we get here, we've exceeded max attempts
    logger.warning(f"Task {id} polling max attempts ({max_attempts}) reached")

    # Get final status before returning
    result = AsyncResult(id, app=celery_app)
    return result
