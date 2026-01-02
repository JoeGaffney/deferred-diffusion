import base64
import io
from pathlib import Path
from typing import List, Literal

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


def text_to_image_call(context: ImageContext) -> List[Path]:
    client = OpenAI()
    model = "gpt-image-1.5"

    result = client.images.generate(
        model=model,
        quality=_quality,
        prompt=context.data.cleaned_prompt,
        size=get_size(context),
    )

    processed_image = process_openai_image_output(result)
    return [context.save_output(processed_image, index=0)]


def image_to_image_call(context: ImageContext) -> List[Path]:
    client = OpenAI()
    model = "gpt-image-1.5"

    # gather all possible reference images
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
        prompt=context.data.cleaned_prompt,
        size=get_size(context),
        image=reference_images,
    )

    processed_image = process_openai_image_output(result)
    return [context.save_output(processed_image, index=0)]


def inpainting_call(context: ImageContext) -> List[Path]:
    client = OpenAI()
    model = "gpt-image-1.5"

    if context.color_image is None:
        raise ValueError("No color image provided")

    if context.mask_image is None:
        raise ValueError("No mask image provided")

    converted_mask = convert_mask_for_inpainting(context.mask_image)

    try:
        result = client.images.edit(
            model=model,
            quality=_quality,
            prompt=context.data.cleaned_prompt,
            size=get_size(context),
            image=convert_pil_to_bytes(context.color_image),
            mask=convert_pil_to_bytes(converted_mask),
        )
    except Exception as e:
        raise ValueError(f"Error in OpenAI inpainting call: {e}")

    processed_image = process_openai_image_output(result)
    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)
    return text_to_image_call(context)
