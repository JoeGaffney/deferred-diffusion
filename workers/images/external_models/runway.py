import time
from typing import Literal

from diffusers.utils import load_image
from PIL import Image
from runwayml import RunwayML
from runwayml.types.text_to_image_create_params import ContentModeration, ReferenceImage

from common.logger import logger
from images.context import ImageContext
from videos.models.runway_gen import pill_to_base64


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


def poll_until_resolved(id, timeout=500, poll_interval=10) -> Image.Image:
    client = RunwayML()

    # Poll the task until it's complete
    time.sleep(poll_interval)
    task = client.tasks.retrieve(id)
    attempts = 0

    start_time = time.time()
    while task.status not in ["SUCCEEDED", "FAILED"]:
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            raise Exception(f"Task polling exceeded timeout ({timeout} seconds)")

        time.sleep(poll_interval)
        remaining_time = timeout - elapsed_time
        logger.info(f"Polling task {id}... {remaining_time:.1f} seconds left")

        task = client.tasks.retrieve(id)
        logger.info(f"Checking Task: {task}")
        attempts += 1

    if task.status == "SUCCEEDED" and task.output and len(task.output) > 0:
        return load_image(task.output[0])

    raise Exception(f"Task failed: {task}")


def text_to_image_call(client: RunwayML, context: ImageContext) -> Image.Image:
    logger.info(f"Text to image call {context.data.model_path}")
    if context.data.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

    try:
        task = client.text_to_image.create(
            model=context.data.model_path,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        )
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    id = task.id
    return poll_until_resolved(id)


def image_to_image_call(client: RunwayML, context: ImageContext) -> Image.Image:
    if context.data.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

    # gather all possible reference images we piggy back on the ipdapter images
    reference_images = []
    if context.color_image:
        reference_images.append(ReferenceImage(uri=f"data:image/png;base64,{pill_to_base64(context.color_image)}"))

    if context.adapters.is_enabled():
        for current in context.adapters.get_images():
            reference_images.append(ReferenceImage(uri=f"data:image/png;base64,{pill_to_base64(current)}"))

    if len(reference_images) == 0:
        raise ValueError("No reference images provided")

    logger.info(f"Image to image call {context.data.model_path} using {len(reference_images)} images")
    try:
        task = client.text_to_image.create(
            model=context.data.model_path,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            reference_images=reference_images,
            seed=context.data.seed,
            # content_moderation=ContentModeration(public_figure_threshold="low"),
        )
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    id = task.id
    return poll_until_resolved(id)


def main(context: ImageContext) -> Image.Image:
    client = RunwayML()

    mode = "img_to_img"
    if context.data.image is None:
        mode = "text_to_image"
    if context.adapters.is_enabled():
        mode = "img_to_img"

    if mode == "text_to_image":
        return text_to_image_call(client, context)
    elif mode == "img_to_img":
        return image_to_image_call(client, context)

    raise ValueError(f"Invalid mode {mode} for RunwayML API")
