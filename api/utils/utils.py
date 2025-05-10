import asyncio
import base64
import gc
import io
import math
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Literal, Optional, Tuple

# import torch
from celery.result import AsyncResult
from fastapi import HTTPException
from PIL import Image

from common.logger import logger

# Resolutions = Literal["1080p", "900p", "720p", "576p", "540p", "480p", "432p", "360p"]
# resolutions_16_9 = {
#     "1080p": (1920, 1080),  # by 8
#     "900p": (1600, 900),
#     "720p": (1280, 720),  # by 8
#     "576p": (1024, 576),  # by 8 and 32
#     "540p": (960, 540),
#     "480p": (854, 480),
#     "432p": (768, 432),  # by 8
#     "360p": (640, 360),
# }


# def get_16_9_resolution(resolution: Resolutions) -> Tuple[int, int]:
#     return resolutions_16_9.get(resolution, (960, 540))


# def get_tmp_dir() -> str:
#     subdir = os.path.join(tempfile.gettempdir(), "deferred-diffusion")
#     os.makedirs(subdir, exist_ok=True)
#     return subdir


# def ensure_path_exists(path):
#     my_dir = os.path.dirname(path)
#     if not os.path.exists(my_dir):
#         try:
#             os.makedirs(my_dir)
#         except Exception as e:
#             print(e)
#             pass


# def save_copy_with_timestamp(path):
#     if os.path.exists(path):
#         directory, filename = os.path.split(path)
#         name, ext = os.path.splitext(filename)

#         # Create the timestamped path
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Keep only 3 digits of milliseconds
#         timestamp_path = os.path.join(directory, "tmp", f"{name}_{timestamp}{ext}")
#         ensure_path_exists(timestamp_path)

#         shutil.copy(path, timestamp_path)


# def ensure_divisible(value: int, divisor=16) -> int:
#     return (value // divisor) * divisor


# def resize_image(image, division=16, scale=1.0, max_width=2048, max_height=2048):
#     """Ensure the new dimensions do not exceed max_width and max_height"""
#     width = min(image.size[0] * scale, max_width)
#     height = min(image.size[1] * scale, max_height)

#     # Adjust width and height to be divisible by 32 or 8
#     width = math.ceil(width / division) * division
#     height = math.ceil(height / division) * division

#     logger.info(f"Image Resized from: {image.size} to {width}x{height}")
#     return image.resize((width, height))


def load_image_from_base64(base64_bytes: str) -> Image.Image:

    try:
        # Convert bytes to a PIL image
        tmp_bytes = base64.b64decode(base64_bytes)
        image = Image.open(io.BytesIO(tmp_bytes))
        image = image.convert("RGB")  # Ensure the image is in RGB mode
        logger.info(f"Image loaded from Base64 bytes, size: {image.size}")
        return image
    except Exception as e:
        raise ValueError(f"Invalid Base64 data: {type(base64_bytes)} {e}") from e


# def load_image_if_exists(base64_bytes: Optional[str]) -> Optional[Image.Image]:
#     """Load image from Base64 string if it exists."""
#     if (base64_bytes is None) or (base64_bytes == ""):
#         return None

#     return load_image_from_base64(base64_bytes)


# def convert_pil_to_bytes(image: Image.Image) -> io.BytesIO:
#     """Convert PIL Image to bytes."""
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format="PNG")
#     img_byte_arr.seek(0)
#     img_byte_arr.name = "image.png"  # Crucial: tells the correct MIME type

#     return img_byte_arr


# def pil_to_base64(image: Image.Image) -> bytes:
#     """Convert PIL Image to base64 encoded bytes."""
#     img_byte_arr = convert_pil_to_bytes(image)
#     return base64.b64encode(img_byte_arr.getvalue())


# def mp4_to_base64(file_path: str) -> bytes:
#     """Convert an MP4 file to base64 encoded bytes."""
#     with open(file_path, "rb") as video_file:
#         return base64.b64encode(video_file.read())


# def convert_mask_for_inpainting(mask: Image.Image) -> Image.Image:
#     """Convert mask image to RGBA format for OpenAI inpainting API.
#     White areas will become transparent (edited), black areas will be preserved."""
#     if mask.mode != "L":
#         mask = mask.convert("L")

#     # Create RGBA where black is opaque (preserved) and white is transparent (edited)
#     rgba = Image.new("RGBA", mask.size, (0, 0, 0, 255))
#     rgba.putalpha(Image.eval(mask, lambda x: 255 - x))
#     return rgba


# def cache_info_decorator(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")

#         result = func(*args, **kwargs)
#         end = time.time()

#         info = func.cache_info()
#         logger.info(
#             f"Cache info - hits: {info.hits}, misses: {info.misses}, "
#             f"current size: {info.currsize}, max size: {info.maxsize}"
#             f" - took {end - start:.2f}s"
#         )
#         return result

#     return wrapper


# def get_gpu_memory_usage():
#     reserved = torch.cuda.memory_reserved() / 1e9
#     allocated = torch.cuda.memory_allocated() / 1e9
#     available, total = torch.cuda.mem_get_info()
#     used = (total - available) / 1e9
#     total = total / 1e9
#     usage_percent = (used / total) * 100

#     return (
#         total,
#         used,
#         reserved,
#         allocated,
#         usage_percent,
#     )


# def get_gpu_memory_usage_pretty():
#     total, used, reserved, allocated, usage_percent = get_gpu_memory_usage()

#     return (
#         f"GPU Memory Usage: {used:.2f}GB / {total:.2f}GB,  "
#         f"Reserved: {reserved:.2f}GB, "
#         f"Allocated: {allocated:.2f}GB, "
#         f"Usage: {usage_percent:.2f}%"
#     )


# def should_free_gpu_memory(threshold_percent: float = 50.0):
#     total, used, reserved, allocated, usage_percent = get_gpu_memory_usage()
#     logger.info(f"{get_gpu_memory_usage_pretty()}")
#     return usage_percent > threshold_percent


# def free_gpu_memory(threshold_percent: float = 50.0):
#     if (should_free_gpu_memory(threshold_percent=threshold_percent) == False) or (torch.cuda.is_available() == False):
#         return

#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()

#     after_stats = get_gpu_memory_usage_pretty()
#     logger.warning(f"Clean {after_stats}")

# from celery import current_app
from worker import celery_app

# async def poll_task_until_complete(task_id, max_polling_time=300, polling_interval=1) -> AsyncResult:
#     """
#     Poll a Celery task until it's complete or until timeout.

#     Args:
#         task_id: The ID of the Celery task to poll
#         max_polling_time: Maximum time to poll in seconds (default: 5 minutes)
#         polling_interval: How often to check task status in seconds
#     """
#     start_time = asyncio.get_event_loop().time()

#     while True:
#         task_result = AsyncResult(task_id)

#         if task_result.ready():
#             if task_result.failed():
#                 raise HTTPException(status_code=500, detail=f"Task processing failed: {task_result.result}")
#             return task_result

#         # Check if we've exceeded max polling time
#         elapsed = asyncio.get_event_loop().time() - start_time
#         if elapsed > max_polling_time:
#             raise HTTPException(status_code=408, detail=f"Task processing timeout after {max_polling_time} seconds")

#         # Wait before polling again
#         await asyncio.sleep(polling_interval)


async def poll_task_until_complete(task_id, max_attempts=30, polling_interval=3) -> AsyncResult:
    """
    Poll a Celery task until it's complete, max attempts reached, or timeout.

    Args:
        task_id: The ID of the Celery task to poll
        max_attempts: Maximum number of polling attempts (default: 30)
        polling_interval: How often to check task status in seconds (default: 1)

    Returns:
        AsyncResult object with task result/status
    """
    # Ensure task_id is a string
    if not isinstance(task_id, str):
        logger.warning(f"task_id is not a string, converting from {type(task_id)}")
        task_id = str(task_id)

    logger.debug(f"Polling task with ID: {task_id}, max attempts: {max_attempts}")

    # Use iteration count instead of time-based approach
    for attempt in range(max_attempts):
        # Get task result
        task_result = celery_app.AsyncResult(task_id)

        # Add attempt number to the result object for debugging
        task_result._attempt = attempt + 1

        if task_result.ready():
            if task_result.failed():
                error_msg = str(task_result.result)
                logger.error(f"Task {task_id} failed: {error_msg}")
                return task_result
            elif task_result.successful():
                logger.info(f"Task {task_id} completed successfully after {attempt + 1} attempts")
                return task_result

        # Wait before polling again
        await asyncio.sleep(polling_interval)

    # If we get here, we've exceeded max attempts
    logger.warning(f"Task {task_id} polling max attempts ({max_attempts}) reached")

    # Get final status before returning
    task_result = celery_app.AsyncResult(task_id)
    task_result._max_attempts_reached = True
    return task_result
