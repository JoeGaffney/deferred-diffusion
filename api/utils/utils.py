import math
import os
import shutil
import time
from datetime import datetime
from typing import Literal, Tuple

import torch
from diffusers.utils import load_image
from utils.logger import logger

Resolutions = Literal["1080p", "900p", "720p", "576p", "540p", "480p", "432p", "360p"]
resolutions_16_9 = {
    "1080p": (1920, 1080),  # by 8
    "900p": (1600, 900),
    "720p": (1280, 720),  # by 8
    "576p": (1024, 576),  # by 8 and 32
    "540p": (960, 540),
    "480p": (854, 480),
    "432p": (768, 432),  # by 8
    "360p": (640, 360),
}


def get_16_9_resolution(resolution: Resolutions) -> Tuple[int, int]:
    return resolutions_16_9.get(resolution, (960, 540))


def ensure_path_exists(path):
    my_dir = os.path.dirname(path)
    if not os.path.exists(my_dir):
        try:
            os.makedirs(my_dir)
        except Exception as e:
            print(e)
            pass


def save_copy_with_timestamp(path):
    if os.path.exists(path):
        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        # Create the timestamped path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Keep only 3 digits of milliseconds
        timestamp_path = os.path.join(directory, "tmp", f"{name}_{timestamp}{ext}")
        ensure_path_exists(timestamp_path)

        shutil.copy(path, timestamp_path)


def resize_image(image, division=16, scale=1.0, max_width=2048, max_height=2048):

    # Ensure the new dimensions do not exceed max_width and max_height
    width = min(image.size[0] * scale, max_width)
    height = min(image.size[1] * scale, max_height)

    # Adjust width and height to be divisible by 32 or 8
    width = math.ceil(width / division) * division
    height = math.ceil(height / division) * division

    logger.info(f"Image Resized from: {image.size} to {width}x{height}")
    return image.resize((width, height))


def load_image_if_exists(image_path):
    if (image_path is None) or (image_path == ""):
        return None

    if not os.path.exists(image_path):
        return None

    image = load_image(image_path)

    logger.info(f"Image loaded from {image_path} size: {image.size}")
    return image


def cache_info_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")

        result = func(*args, **kwargs)
        end = time.time()

        info = func.cache_info()
        logger.info(
            f"Cache info - hits: {info.hits}, misses: {info.misses}, "
            f"current size: {info.currsize}, max size: {info.maxsize}"
            f" - took {end - start:.2f}s"
        )
        return result

    return wrapper


def get_gpu_memory_usage():
    reserved = torch.cuda.memory_reserved() / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    available, total = torch.cuda.mem_get_info()
    used = (total - available) / 1e9
    total = total / 1e9
    usage_percent = (used / total) * 100

    return (
        total,
        used,
        reserved,
        allocated,
        usage_percent,
    )


def get_gpu_memory_usage_pretty():
    total, used, reserved, allocated, usage_percent = get_gpu_memory_usage()

    return (
        f"GPU Memory Usage: {used:.2f}GB / {total:.2f}GB,  "
        f"Reserved: {reserved:.2f}GB, "
        f"Allocated: {allocated:.2f}GB, "
        f"Usage: {usage_percent:.2f}%"
    )


def should_free_gpu_memory(threshold_percent: float = 80.0):
    total, used, reserved, allocated, usage_percent = get_gpu_memory_usage()
    logger.info(f"{get_gpu_memory_usage_pretty()}")
    return usage_percent > threshold_percent


def free_gpu_memory():
    if (should_free_gpu_memory(threshold_percent=80.0) == False) or (torch.cuda.is_available() == False):
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    after_stats = get_gpu_memory_usage_pretty()
    logger.warning(f"GPU Memory Clean:\n{after_stats}")
