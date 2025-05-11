import base64
import io
from typing import Literal

from openai import OpenAI
from PIL import Image

from common.logger import logger
from images.context import ImageContext
from utils.utils import convert_mask_for_inpainting, convert_pil_to_bytes


def get_size(
    context: ImageContext,
) -> Literal["1024x1024", "1536x1024", "1024x1536"]:
    size = "1024x1024"
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        size = "1536x1024"
    elif dimension_type == "portrait":
        size = "1024x1536"
    return size


def text_to_image_call(client: OpenAI, context: ImageContext):
    logger.info(f"Text to image call {context.model_config.model_path}")
    result = client.images.generate(
        model=context.model_config.model_path,
        quality=context.get_quality(),
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
    return processed_image


def image_to_image_call(client: OpenAI, context: ImageContext):

    # gather all possible reference images we piggy back on the ipdapter images
    reference_images = []
    if context.color_image:
        reference_images.append(convert_pil_to_bytes(context.color_image))

    if context.ip_adapters_enabled:
        for current in context.get_ip_adapter_images():
            reference_images.append(convert_pil_to_bytes(current))

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    logger.info(f"Image to image call {context.model_config.model_path} using {len(reference_images)} images")

    result = client.images.edit(
        model=context.model_config.model_path,
        quality=context.get_quality(),
        prompt=context.data.prompt,
        size=get_size(context),  # type: ignore type hints wrong
        image=reference_images,
    )
    if result.data is None:
        raise ValueError("No image data returned from OpenAI API")

    result_data = result.data[0]
    image_base64 = result_data.b64_json
    if image_base64 is None:
        raise ValueError("No image data returned from OpenAI API")

    image_bytes = base64.b64decode(image_base64)
    processed_image = Image.open(io.BytesIO(image_bytes))
    return processed_image


def inpainting_call(client: OpenAI, context: ImageContext):
    logger.info(f"Image to image inpainting call {context.model_config.model_path}")
    if context.color_image is None:
        raise ValueError("No color image provided")

    if context.mask_image is None:
        raise ValueError("No mask image provided")

    converted_mask = convert_mask_for_inpainting(context.mask_image)

    result = client.images.edit(
        model=context.model_config.model_path,
        quality=context.get_quality(),
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
    return processed_image


def main(
    context: ImageContext,
    mode="text_to_image",
) -> Image.Image:

    client = OpenAI()

    if mode == "text_to_image":
        if context.ip_adapters_enabled:
            return image_to_image_call(client, context)

        return text_to_image_call(client, context)
    elif mode == "img_to_img":
        return image_to_image_call(client, context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(client, context)

    raise ValueError(f"Invalid mode {mode} for OpenAI API")
