from typing import Literal

from runwayml import RunwayML
from runwayml.types.image_to_video_create_params import (
    Gen4TurboContentModeration as ContentModeration,
)

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
    image = context.image
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    video = context.data.video
    if video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    # TODO switch to closest resolution as per the aspect ratio
    image_uri = f"data:image/png;base64,{pill_to_base64(image)}"
    video_uri = f"data:video/mp4;base64,{video}"

    try:
        task = client.character_performance.create(
            model="act_two",
            character={
                "type": "image",
                "uri": image_uri,
            },
            body_control=True,
            reference={"type": "video", "uri": video_uri},
            ratio=get_aspect_ratio(context),
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
