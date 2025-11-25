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


def video_to_video(context: VideoContext):
    if context.data.video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    client = RunwayML()

    model: Literal["gen4_aleph"] = "gen4_aleph"
    ratio: Literal["1280:720", "720:1280", "960:960"] = "1280:720"
    video_uri = f"data:video/mp4;base64,{context.get_compressed_video()}"
    references = []
    if context.image:
        image_uri = f"data:image/png;base64,{pill_to_base64(context.image)}"
        references.append({"type": "image", "uri": image_uri})
        ratio = get_aspect_ratio(context)

    try:
        task = client.video_to_video.create(
            model=model,
            video_uri=video_uri,
            prompt_text=context.data.cleaned_prompt,
            references=references,  # type: ignore
            ratio=ratio,
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")


def main(context: VideoContext):
    if context.data.video:
        return video_to_video(context)

    if context.image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    client = RunwayML()

    model: Literal["gen4_turbo"] = "gen4_turbo"
    ratio = get_aspect_ratio(context)
    duration: Literal[5, 10] = 5 if context.duration_in_seconds() <= 5 else 10
    image_uri = f"data:image/png;base64,{pill_to_base64(context.image)}"

    try:
        task = client.image_to_video.create(
            model=model,
            prompt_image=image_uri,
            prompt_text=context.data.cleaned_prompt,
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
