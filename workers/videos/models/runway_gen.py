import base64
import time
from io import BytesIO
from typing import Literal

from runwayml import RunwayML

from common.logger import logger
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext


def pill_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image


def poll_until_resolved(context: VideoContext, id, timeout=500, poll_interval=10):
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

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")


def create(context: VideoContext):
    client = RunwayML()

    # atm only these 16 by 9 or 9 by 16 supported
    # TODO switch to closest resolution as per the aspect ratio
    width, height = get_16_9_resolution("1080p")
    image = context.image
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    image = resize_image(image, 1, 1.0, width, height)

    model: Literal["gen3a_turbo", "gen4_turbo"] = "gen3a_turbo"
    ratio: Literal["1280:768", "1280:720", "768:1280"] = "1280:768"
    duration: Literal[5, 10] = 5
    if context.model == "runway/gen4_turbo":
        model = "gen4_turbo"
        ratio = "1280:720"

    # encode image to base64
    base64_image = pill_to_base64(image)

    # Create a new image-to-video task using the "gen3a_turbo" model
    logger.info(f"Creating Runway {model} task")

    task = client.image_to_video.create(
        model=model,
        prompt_image=f"data:image/png;base64,{base64_image}",
        prompt_text=context.data.prompt,
        ratio=ratio,
        duration=duration,
        seed=context.data.seed,
    )
    id = task.id

    logger.info(f"Task ID: {id}")
    return id


def main(context: VideoContext):
    id = create(context)
    return poll_until_resolved(context, id)
