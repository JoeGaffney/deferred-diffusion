from typing import Literal

from diffusers.utils import load_image
from PIL import Image, ImageOps
from runwayml import RunwayML
from runwayml.types.text_to_image_create_params import ContentModeration, ReferenceImage

from common.logger import logger
from images.context import ImageContext
from utils.utils import pill_to_base64


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


def get_size(
    context: ImageContext,
) -> Literal["1024:1024", "1920:1080", "1080:1920"]:
    size = "1024:1024"
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        size = "1920:1080"
    elif dimension_type == "portrait":
        size = "1080:1920"
    return size


def text_to_image_call(context: ImageContext) -> Image.Image:
    client = RunwayML()
    if context.data.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

    try:
        task = client.text_to_image.create(
            model=context.data.model_path,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output and len(task.output) > 0:
        return load_image(task.output[0])

    raise Exception(f"Task failed: {task}")


def image_to_image_call(context: ImageContext) -> Image.Image:
    client = RunwayML()

    if context.data.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

    # This is cheaper for image to image calls
    model = "gen4_image_turbo"

    # gather all possible reference images we piggy back on the ipdapter images
    reference_images = []
    if context.color_image:
        fixed_image = fix_aspect_ratio(context.color_image)
        reference_images.append(ReferenceImage(uri=f"data:image/png;base64,{pill_to_base64(fixed_image)}"))

    for current in context.get_reference_images():
        fixed_image = fix_aspect_ratio(current)
        reference_images.append(ReferenceImage(uri=f"data:image/png;base64,{pill_to_base64(fixed_image)}"))

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    try:
        task = client.text_to_image.create(
            model=model,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            reference_images=reference_images,
            seed=context.data.seed,
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output and len(task.output) > 0:
        return load_image(task.output[0])

    raise Exception(f"Task failed: {task}")


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        if context.get_reference_images() != []:
            return image_to_image_call(context)
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)

    raise ValueError(f"Invalid mode {mode} for RunwayML API")
