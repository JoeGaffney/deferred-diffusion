from pathlib import Path
from typing import List

from PIL import Image, ImageOps

from common.replicate_helpers import process_replicate_image_output, replicate_run
from images.context import ImageContext
from utils.utils import convert_pil_to_bytes


def fix_aspect_ratio(img: Image.Image, safety_margin: float = 0.02) -> Image.Image:
    """
    Adjusts image to meet RunwayML API requirements for aspect ratio (between 0.5 and 2.0).
    Uses ImageOps.fit to center the image when cropping.
    """
    aspect_ratio = img.width / img.height

    # Safe boundaries with margin
    min_ratio = 0.5 + safety_margin
    max_ratio = 2.0 - safety_margin

    # Return original if already within safe range
    if min_ratio <= aspect_ratio <= max_ratio:
        return img

    # Too wide - adjust to safe max ratio
    if aspect_ratio > max_ratio:
        new_height = int(img.width / max_ratio)
        return ImageOps.fit(img, (img.width, new_height))

    # Too tall - adjust to safe min ratio
    new_width = int(img.height * min_ratio)
    return ImageOps.fit(img, (new_width, img.height))


def get_aspect_ratio(context: ImageContext) -> str:
    """
    Convert context dimensions to Runway aspect ratio format.
    """
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "16:9"
    elif dimension_type == "portrait":
        return "9:16"
    else:
        return "1:1"


def text_to_image_call(context: ImageContext) -> List[Path]:
    payload = {
        "prompt": context.data.cleaned_prompt,
        "aspect_ratio": get_aspect_ratio(context),
    }

    output = replicate_run("runwayml/gen4-image", payload)
    processed_image = process_replicate_image_output(output)
    return [context.save_output(processed_image, index=0)]


def image_to_image_call(context: ImageContext) -> List[Path]:
    # Gather all possible reference images
    reference_images = []
    reference_tags = []

    if context.color_image:
        fixed_image = fix_aspect_ratio(context.color_image)
        reference_images.append(convert_pil_to_bytes(fixed_image))
        reference_tags.append("image1")

    for i, current in enumerate(context.get_reference_images()):
        if current is not None:
            fixed_image = fix_aspect_ratio(current)
            reference_images.append(convert_pil_to_bytes(fixed_image))
            reference_tags.append(f"image{i+2}")

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    payload = {
        "prompt": context.data.cleaned_prompt,
        "aspect_ratio": get_aspect_ratio(context),
        "reference_images": reference_images,
        "reference_tags": reference_tags,
    }

    output = replicate_run("runwayml/gen4-image-turbo", payload)
    processed_image = process_replicate_image_output(output)
    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    if context.get_reference_images() != [] or context.color_image:
        return image_to_image_call(context)

    return text_to_image_call(context)
