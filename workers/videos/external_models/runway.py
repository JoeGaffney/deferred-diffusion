from typing import Literal

from runwayml import RunwayML
from runwayml.types.text_to_image_create_params import ContentModeration

from utils.utils import pill_to_base64
from videos.context import VideoContext


def get_aspect_ratio(context: VideoContext) -> Literal["1280:720", "720:1280", "960:960"]:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "1280:720"
    elif dimension_type == "portrait":
        return "720:1280"
    return "960:960"


def main(context: VideoContext):
    client = RunwayML()
    model = "gen4_turbo"

    image = context.image
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    ratio = get_aspect_ratio(context)
    duration: Literal[5, 10] = 5

    # encode image to base64
    image_uri = f"data:image/png;base64,{pill_to_base64(image)}"

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
