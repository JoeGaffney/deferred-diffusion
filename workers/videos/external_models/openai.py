from typing import Literal

import openai
from openai import Omit, OpenAI
from openai.types import VideoSeconds, VideoSize

from common.logger import logger
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def get_aspect_ratio(context: VideoContext) -> VideoSize:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "1280x720"
    elif dimension_type == "portrait":
        return "720x1280"
    return "1280x720"


def main(context: VideoContext):
    client = OpenAI()

    # model = "sora-2-pro"
    model = "sora-2"
    reference_image = None
    if context.image:
        reference_image = convert_pil_to_bytes(context.image)

    try:
        video = client.videos.create_and_poll(
            model=model,
            prompt=context.data.prompt,
            input_reference=reference_image or Omit(),
            size=get_aspect_ratio(context),
            seconds="8" if context.long_video() else "4",
            timeout=60 * 10,  # 10 minutes
        )

        if video.status not in ("succeeded", "completed"):
            print(video)
            raise ValueError(f"Video generation failed with status: {video.status} {str(video.error)}")
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {e}")

    tmp_path = context.tmp_video_path()
    try:
        content = client.videos.download_content(video.id, variant="video")
        content.write_to_file(tmp_path)
    except Exception as e:
        raise RuntimeError(f"Error downloading video content: {e}")

    logger.info(f"Video saved at {tmp_path}")
    return tmp_path
