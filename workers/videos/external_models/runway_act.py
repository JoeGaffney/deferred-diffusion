from typing import Literal, cast

from runwayml import RunwayML
from runwayml.types.text_to_image_create_params import ContentModeration

from common.logger import logger
from utils.utils import pill_to_base64, resize_image
from videos.context import VideoContext


def main(context: VideoContext):
    client = RunwayML()
    # Only 'act_two' model is supported for this context
    if context.data.model_path not in ["act_two"]:
        raise ValueError("Only 'act_two' model is supported")

    image = context.image
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    video = context.data.video
    if video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    # TODO switch to closest resolution as per the aspect ratio
    image = resize_image(image, 1, 1.0, 2048, 2048)
    model: Literal["act_two"] = cast(Literal["act_two"], context.data.model_path)
    ratio: Literal["1280:720", "720:1280", "960:960"] = "1280:720"

    logger.info(f"Creating Runway {model} task")
    try:
        task = client.character_performance.create(
            model=model,
            character={
                "type": "image",
                "uri": f"data:image/png;base64,{pill_to_base64(image)}",
            },
            body_control=True,
            reference={"type": "video", "uri": f"data:video/mp4;base64,{video}"},
            ratio=ratio,
            seed=context.data.seed,
            content_moderation=ContentModeration(public_figure_threshold="low"),
        ).wait_for_task_output(
            timeout=60 * 60
        )  # 60 minutes timeout
    except Exception as e:
        raise RuntimeError(f"Error calling RunwayML API: {e}")

    if task.status == "SUCCEEDED" and task.output:
        return context.save_video_url(task.output[0])

    raise Exception(f"Task failed: {task}")
