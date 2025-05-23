import base64
import io
import time
from typing import Literal

from PIL import Image
from runwayml import RunwayML

from common.logger import logger
from images.context import ImageContext
from utils.utils import convert_mask_for_inpainting, convert_pil_to_bytes


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


def poll_result(id, wait=10, max_attempts=30):
    client = RunwayML()

    # Poll the task until it's complete
    time.sleep(wait)
    task = client.tasks.retrieve(id)
    attempts = 0

    while task.status not in ["SUCCEEDED", "FAILED"]:
        if attempts >= max_attempts:
            raise Exception(f"Task polling exceeded maximum attempts ({max_attempts})")

        time.sleep(wait)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(id)
        logger.info(f"Checking Task: {task}")
        attempts += 1

    if task.status == "SUCCEEDED" and task.output:
        image_bytes = base64.b64decode(task.output[0])
        processed_image = Image.open(io.BytesIO(image_bytes))
        return processed_image

    raise Exception(f"Task failed: {task}")


def text_to_image_call(client: RunwayML, context: ImageContext):
    logger.info(f"Text to image call {context.model_config.model_path}")
    if context.model_config.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

    try:
        task = client.text_to_image.create(
            model=context.model_config.model_path,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            seed=context.data.seed,
        )
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    id = task.id
    return poll_result(id)


def image_to_image_call(client: RunwayML, context: ImageContext):
    if context.model_config.model_path != "gen4_image":
        raise ValueError("Only gen4_image model is supported")

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

    try:
        task = client.text_to_image.create(
            model=context.model_config.model_path,
            prompt_text=context.data.prompt,
            ratio=get_size(context),
            reference_images=reference_images,
            seed=context.data.seed,
        )
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    id = task.id
    return poll_result(id)


def main(
    context: ImageContext,
    mode="text_to_image",
) -> Image.Image:

    client = RunwayML()

    if mode == "text_to_image":
        if context.ip_adapters_enabled:
            return image_to_image_call(client, context)

        return text_to_image_call(client, context)
    elif mode == "img_to_img":
        return image_to_image_call(client, context)
    elif mode == "img_to_img_inpainting":
        return image_to_image_call(client, context)

    raise ValueError(f"Invalid mode {mode} for RunwayML API")
