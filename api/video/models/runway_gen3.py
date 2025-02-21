from io import BytesIO
import time, base64
from typing import Literal
from utils.utils import get_16_9_resolution
from common.context import Context
from runwayml import RunwayML


def poll_result(context: Context, task_id, wait=10):
    client = RunwayML()

    # Poll the task until it's complete
    time.sleep(wait)  # Wait for a second before polling
    task = client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        time.sleep(wait)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(task_id)
        context.log(f"Checking Task: {task}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")


def pill_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")
    return base64_image


def create(context: Context):
    client = RunwayML()

    # atm only these 16 by 9 or 9 by 16 supported
    # TODO switch to closest resolution as per the aspect ratio
    width, height = get_16_9_resolution("1080p")
    context.max_height = 10000
    context.max_width = 10000
    image = context.load_image(division=1)
    image = image.resize((width, height))

    ratio: list[Literal["1280:768", "768:1280"]] = ["1280:768", "768:1280"]
    duration: list[Literal[5, 10]] = [5, 10]

    # encode image to base64
    base64_image = pill_to_base64(image)
    model: Literal["gen3a_turbo"] = "gen3a_turbo"

    # Create a new image-to-video task using the "gen3a_turbo" model
    context.log(f"Creating Runway {model} task")

    task = client.image_to_video.create(
        model=model,
        prompt_image=f"data:image/png;base64,{base64_image}",
        prompt_text=context.prompt,
        ratio=ratio[0],
        duration=duration[0],
        # seed=context.seed,
    )
    task_id = task.id

    # context.task_id = task_id
    context.log(f"Task ID: {task_id}")
    return task_id


def main(context: Context):
    task_id = create(context)
    return poll_result(context, task_id)


if __name__ == "__main__":
    context = Context(
        input_image_path="../tmp/tornado_v001.jpg",
        output_video_path="../tmp/output/tornado_v001_runway_video.mp4",
        strength=0.2,
        prompt="A farm landscape with a tornado destroying houses and ripping up land,Fire and lighting",
        num_inference_steps=50,
    )
    context = Context(
        input_image_path="../tmp/earth_quake_v001.JPG",
        output_video_path="../tmp/output/earth_quake_runway_video.mp4",
        strength=0.2,
        prompt="Ground collaping trees falling, Dust and debris, Static camera, DLSR photo",
        num_inference_steps=50,
    )
    main(context)
    # task_id = create(context)
    # poll_result(context, task_id)
