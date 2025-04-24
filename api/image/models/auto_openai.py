import base64
import io
from functools import lru_cache
from typing import Literal

from openai import OpenAI
from PIL import Image

from common.logger import logger
from common.pipeline_helpers import optimize_pipeline
from image.context import ImageContext
from image.models.diffusers_helpers import (
    image_to_image_call,
    inpainting_call,
    text_to_image_call,
)
from image.schemas import PipelineConfig
from utils.utils import cache_info_decorator

client = OpenAI()


def convert_pil_to_bytes(image: Image.Image) -> io.BytesIO:
    """Convert PIL Image to bytes for OpenAI API."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_byte_arr.name = "image.png"  # Crucial: tells OpenAI the correct MIME type

    return img_byte_arr


def convert_mask_for_inpainting(mask: Image.Image) -> Image.Image:
    """Convert mask image to RGBA format for OpenAI inpainting API.
    White areas will become transparent (edited), black areas will be preserved."""
    if mask.mode != "L":
        mask = mask.convert("L")

    # Create RGBA where black is opaque (preserved) and white is transparent (edited)
    rgba = Image.new("RGBA", mask.size, (0, 0, 0, 255))
    rgba.putalpha(Image.eval(mask, lambda x: 255 - x))
    return rgba


def get_size(
    context: ImageContext,
) -> Literal["1024x1024", "1536x1024", "1024x1536"]:
    size = "1024x1024"
    if context.dimension_type == "landscape":
        size = "1536x1024"
    elif context.dimension_type == "portrait":
        size = "1024x1536"
    return size


def text_to_image_call(context: ImageContext):
    logger.info(f"Text to image call {context.model_config.model_path}")
    result = client.images.generate(
        model=context.model_config.model_path,
        quality=context.quality,
        prompt=context.data.prompt,
        size=get_size(context),
    )
    if result.data is None:
        raise ValueError("No image data returned from OpenAI API")

    result_data = result.data[0]
    image_base64 = result_data.b64_json
    if image_base64 is None:
        raise ValueError("No image data returned from OpenAI API")

    image_bytes = base64.b64decode(image_base64)
    processed_image = Image.open(io.BytesIO(image_bytes))

    # not sure if we should resize the image again??
    # processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def image_to_image_call(context: ImageContext):
    logger.info(f"Image to image call {context.model_config.model_path}")
    if context.color_image is None:
        raise ValueError("No color image provided")

    result = client.images.edit(
        model=context.model_config.model_path,
        quality=context.quality,
        prompt=context.data.prompt,
        size=get_size(context),  # type: ignore type hints wrong
        image=[convert_pil_to_bytes(context.color_image)],
    )
    if result.data is None:
        raise ValueError("No image data returned from OpenAI API")

    result_data = result.data[0]
    image_base64 = result_data.b64_json
    if image_base64 is None:
        raise ValueError("No image data returned from OpenAI API")

    image_bytes = base64.b64decode(image_base64)
    processed_image = Image.open(io.BytesIO(image_bytes))

    # not sure if we should resize the image again??
    # processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def inpainting_call(context: ImageContext):
    logger.info(f"Image to image inpainting call {context.model_config.model_path}")
    if context.color_image is None:
        raise ValueError("No color image provided")

    if context.mask_image is None:
        raise ValueError("No mask image provided")

    converted_mask = convert_mask_for_inpainting(context.mask_image)

    result = client.images.edit(
        model=context.model_config.model_path,
        quality=context.quality,
        prompt=context.data.prompt,
        size=get_size(context),  # type: ignore type hints wrong
        image=[convert_pil_to_bytes(context.color_image)],
        mask=convert_pil_to_bytes(converted_mask),
    )
    if result.data is None:
        raise ValueError("No image data returned from OpenAI API")

    result_data = result.data[0]
    image_base64 = result_data.b64_json
    if image_base64 is None:
        raise ValueError("No image data returned from OpenAI API")

    image_bytes = base64.b64decode(image_base64)
    processed_image = Image.open(io.BytesIO(image_bytes))

    # not sure if we should resize the image again??
    # processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def main(
    context: ImageContext,
    mode="text",
):
    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    return "invalid mode"
