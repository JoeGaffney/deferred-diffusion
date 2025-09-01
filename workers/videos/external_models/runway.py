from typing import Literal, cast

from runwayml import RunwayML
from runwayml.types.text_to_image_create_params import ContentModeration

from common.logger import logger
from utils.utils import pill_to_base64, resize_image
from videos.context import VideoContext


def main(context: VideoContext):
    client = RunwayML()
    model = "gen4_turbo"
    if context.data.model == "runway-gen-3":
        model = "gen3a_turbo"

    # atm only these 16 by 9 or 9 by 16 supported
    # TODO switch to closest resolution as per the aspect ratio
    image = context.image
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    image = resize_image(image, 1, 1.0, 2048, 2048)

    model: Literal["gen3a_turbo", "gen4_turbo"] = cast(Literal["gen3a_turbo", "gen4_turbo"], context.data.model_path)
    ratio: Literal["1280:768", "1280:720", "768:1280"] = "1280:768"
    duration: Literal[5, 10] = 5
    if model == "gen4_turbo":
        ratio = "1280:720"

    # encode image to base64
    image_uri = f"data:image/png;base64,{pill_to_base64(image)}"

    # Create a new image-to-video task using the "gen3a_turbo" model
    logger.info(f"Creating Runway {model} task")

    try:
        task = client.image_to_video.create(
            model=model,
            prompt_image=image_uri,
            prompt_text=context.data.prompt,
            ratio=ratio,
            duration=duration,
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
