from typing import Literal, cast

from runwayml import RunwayML

from common.logger import logger
from videos.context import VideoContext


def main(context: VideoContext):
    client = RunwayML()
    if context.data.model_path not in ["upscale_v1"]:
        raise ValueError("Only 'upscale_v1' model is supported")

    video = context.data.video
    if video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    model: Literal["upscale_v1"] = cast(Literal["upscale_v1"], context.data.model_path)

    logger.info(f"Creating Runway {model} task")
    video = video.replace("\n", "").replace("\r", "")
    video_uri = f"data:video/mp4;base64,{video}"

    try:
        task = client.video_upscale.create(
            model=model,
            video_uri=video_uri,
        ).wait_for_task_output()
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
