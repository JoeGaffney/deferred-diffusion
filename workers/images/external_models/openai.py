import base64
import io
from typing import Literal

from openai import OpenAI
from openai.types.images_response import ImagesResponse
from PIL import Image

from images.context import ImageContext
from utils.utils import convert_mask_for_inpainting, convert_pil_to_bytes

_quality: Literal["high", "auto"] = "auto"


def get_size(
    context: ImageContext,
) -> Literal["1024x1024", "1536x1024", "1024x1536"]:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "1536x1024"
    elif dimension_type == "portrait":
        return "1024x1536"
    return "1024x1024"


def process_openai_image_output(output: ImagesResponse) -> Image.Image:
    if output.data is None or len(output.data) == 0:
        raise ValueError("No image data returned from OpenAI API")

    result_data = output.data[0]
    image_base64 = result_data.b64_json
    if image_base64 is None:
        raise ValueError("No image data returned from OpenAI API")

    try:
        image_bytes = base64.b64decode(image_base64)
        processed_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error processing image data from OpenAI API: {e}")

    return processed_image


def text_to_image_call(context: ImageContext):
    client = OpenAI()
    model = "gpt-image-1" if context.data.high_quality else "gpt-image-1-mini"

    result = client.images.generate(
        model=model,
        quality=_quality,
        prompt=context.data.prompt,
        size=get_size(context),
    )

    return process_openai_image_output(result)


def image_to_image_call(context: ImageContext):
    client = OpenAI()
    model = "gpt-image-1" if context.data.high_quality else "gpt-image-1-mini"

    # gather all possible reference images we piggy back on the ipdapter images
    reference_images = []
    if context.color_image:
        reference_images.append(convert_pil_to_bytes(context.color_image))

    for current in context.get_reference_images():
        if current is not None:
            reference_images.append(convert_pil_to_bytes(current))

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    result = client.images.edit(
        model=model,
        quality=_quality,
        prompt=context.data.prompt,
        size=get_size(context),
        image=reference_images,
    )

    return process_openai_image_output(result)


def inpainting_call(context: ImageContext):
    client = OpenAI()
    model = "gpt-image-1-mini"

    if context.color_image is None:
        raise ValueError("No color image provided")

    if context.mask_image is None:
        raise ValueError("No mask image provided")

    converted_mask = convert_mask_for_inpainting(context.mask_image)

    try:
        result = client.images.edit(
            model=model,
            quality=_quality,
            prompt=context.data.prompt,
            size=get_size(context),  # type: ignore
            image=[convert_pil_to_bytes(context.color_image)],
            mask=convert_pil_to_bytes(converted_mask),
        )
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {e}")

    return process_openai_image_output(result)


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        if len(context.get_reference_images()) > 0:
            return image_to_image_call(context)
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Invalid mode {mode} for OpenAI API")
