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

    video = context.data.video
    if video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    ratio: Literal["1280:720", "720:1280", "960:960"] = "1280:720"
    video_uri = f"data:video/mp4;base64,{video}"

    references = []
    image = context.image
    if image:
        image_uri = f"data:image/png;base64,{pill_to_base64(image)}"
        references.append({"type": "image", "uri": image_uri})
        ratio = get_aspect_ratio(context)

    try:
        task = client.video_to_video.create(
            model="gen4_aleph",
            video_uri=video_uri,
            prompt_text=context.data.prompt,
            references=references,
            ratio=ratio,
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
