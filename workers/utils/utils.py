import base64
import io
import math
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Literal, Optional, Tuple

from diffusers.utils import export_to_video, load_video
from PIL import Image

from common.logger import logger

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


def time_info_decorator(func):
    def wrapper(*args, **kwargs):

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        if elapsed > 1.0:  # Only log if execution took more than 1.0 second
            args_str = str(args)[:100] + ("..." if len(str(args)) > 100 else "")
            kwargs_str = str(kwargs)[:100] + ("..." if len(str(kwargs)) > 100 else "")

            info = f"Calling {func.__name__} with args: {args_str}, kwargs: {kwargs_str}"
            logger.info(f"{info}, took: {elapsed:.2f}s")

        return result

    return wrapper


def get_16_9_resolution(resolution: Resolutions) -> Tuple[int, int]:
    return resolutions_16_9.get(resolution, (960, 540))


def get_tmp_dir(model="") -> str:
    subdir = os.path.join(tempfile.gettempdir(), "deferred-diffusion", "workers")
    if model:
        subdir = os.path.join(subdir, model)

    os.makedirs(subdir, exist_ok=True)
    return subdir


def ensure_path_exists(path):
    my_dir = os.path.dirname(path)
    if not os.path.exists(my_dir):
        try:
            os.makedirs(my_dir)
        except Exception as e:
            print(e)
            pass


def ensure_divisible(value: int, divisor=16) -> int:
    if divisor <= 1:
        return value

    return (value // divisor) * divisor


def resize_image(image, division=16, scale=1.0, max_width=2048, max_height=2048) -> Image.Image:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Scale original size
    target_width = orig_width * scale
    target_height = orig_height * scale

    # simple rounding to nearest divisible
    width = int(ensure_divisible(min(target_width, max_width), division))
    height = int(ensure_divisible(min(target_height, max_height), division))

    # # Fit inside max bounds while preserving aspect ratio
    # scale_factor = min(max_width / target_width, max_height / target_height, 1.0)
    # target_width *= scale_factor
    # target_height *= scale_factor

    # # Round BOTH dimensions down to divisible sizes
    # width = int(math.floor(target_width / division) * division)
    # height = int(math.floor(target_height / division) * division)

    # Final check: skip resize if not needed
    if (width, height) == image.size:
        logger.info(f"No resize needed. Image size is already {width}x{height} (div/{division})")
        return image

    logger.info(f"Resizing from {image.size} to ({width}, {height}) (div/{division})")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_image_from_base64(base64_bytes: str) -> Image.Image:

    try:
        # Convert bytes to a PIL image
        tmp_bytes = base64.b64decode(base64_bytes)
        image = Image.open(io.BytesIO(tmp_bytes))
        image = image.convert("RGB")  # type: ignore
        logger.info(f"Image loaded from Base64 bytes, size: {image.size}")
        return image
    except Exception as e:
        raise ValueError(f"Invalid Base64 data: {type(base64_bytes)} {e}") from e


def load_image_if_exists(base64_bytes: Optional[str]) -> Optional[Image.Image]:
    """Load image from Base64 string if it exists."""
    if (base64_bytes is None) or (base64_bytes == ""):
        return None

    return load_image_from_base64(base64_bytes)


def convert_pil_to_bytes(image: Image.Image) -> io.BytesIO:
    """Convert PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_byte_arr.name = "image.png"  # Crucial: tells the correct MIME type

    return img_byte_arr


def pil_to_base64(image: Image.Image) -> bytes:
    """Convert PIL Image to base64 encoded bytes."""
    img_byte_arr = convert_pil_to_bytes(image)
    return base64.b64encode(img_byte_arr.getvalue())


def mp4_to_base64(file_path: str) -> bytes:
    """Convert an MP4 file to base64 encoded bytes."""
    with open(file_path, "rb") as video_file:
        return base64.b64encode(video_file.read())


def pill_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image


def convert_mask_for_inpainting(mask: Image.Image) -> Image.Image:
    """Convert mask image to RGBA format for OpenAI inpainting API.
    White areas will become transparent (edited), black areas will be preserved."""
    if mask.mode != "L":
        mask = mask.convert("L")

    # Create RGBA where black is opaque (preserved) and white is transparent (edited)
    rgba = Image.new("RGBA", mask.size, (0, 0, 0, 255))
    rgba.putalpha(Image.eval(mask, lambda x: 255 - x))
    return rgba


def load_video_bytes_if_exists(base64_bytes: Optional[str]) -> Optional[bytes]:
    """Load video from Base64 string if it exists, return raw bytes."""
    if (base64_bytes is None) or (base64_bytes == ""):
        return None
    try:
        return base64.b64decode(base64_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Base64 video data: {type(base64_bytes)} {e}") from e


def load_video_frames_if_exists(base64_bytes: Optional[str]) -> Optional[list[Image.Image]]:
    """Load video from Base64 string and return frames as PIL images."""
    video_bytes = load_video_bytes_if_exists(base64_bytes)
    if video_bytes is None:
        return None

    video_path = tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".mp4").name
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    pil_images = load_video(video_path)
    return pil_images


def load_video_into_file(base64_bytes: Optional[str]) -> str | None:
    """Load video from Base64 string and return the file path."""
    video_bytes = load_video_bytes_if_exists(base64_bytes)
    if video_bytes is None:
        return None

    video_path = tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".mp4").name
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    return video_path
