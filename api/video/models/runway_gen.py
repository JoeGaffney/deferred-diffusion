import base64
import time
from io import BytesIO
from typing import Literal

from runwayml import RunwayML

from utils.logger import logger
from utils.utils import get_16_9_resolution
from video.context import VideoContext
from video.schemas import VideoRequest


def poll_result(context: VideoContext, task_id, wait=10):
    client = RunwayML()

    # Poll the task until it's complete
    time.sleep(wait)  # Wait for a second before polling
    task = client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        time.sleep(wait)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(task_id)
        logger.info(f"Checking Task: {task}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")


def pill_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image


def create(context: VideoContext):
    client = RunwayML()

    # atm only these 16 by 9 or 9 by 16 supported
    # TODO switch to closest resolution as per the aspect ratio
    width, height = get_16_9_resolution("1080p")
    context.data.max_height = 10000
    context.data.max_width = 10000
    image = context.load_image(division=1)
    image = image.resize((width, height))

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
        # seed=context.seed,
    )
    task_id = task.id

    # context.task_id = task_id
    logger.info(f"Task ID: {task_id}")
    return task_id


def main(context: VideoContext):
    task_id = create(context)
    return poll_result(context, task_id)


if __name__ == "__main__":
    context = VideoContext(
        VideoRequest(
            model="runway/gen3a_turbo",
            input_image_path="../tmp/color_v001.jpeg",
            output_video_path="../tmp/output/runway_video.mp4",
            strength=0.2,
            prompt="A farm landscape with a tornado destroying houses and ripping up land,Fire and lighting",
            num_inference_steps=50,
        )
    )

    main(context)
