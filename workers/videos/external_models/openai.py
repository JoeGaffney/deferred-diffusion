from typing import Literal

import openai
from openai import Omit, OpenAI
from openai.types import VideoSeconds, VideoSize
from PIL import Image

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


def resize_image_to_aspect_ratio(image, context: VideoContext) -> Image.Image:
    dimension_type = context.get_dimension_type()
    if dimension_type == "portrait":
        return image.resize((720, 1280))

    # default to landscape
    return image.resize((1280, 720))


def main(context: VideoContext):
    client = OpenAI()

    # NOTE base model seems a bit crap but is 3 times cheaper than pro
    model: Literal["sora-2", "sora-2-pro"] = "sora-2"
    if context.data.high_quality:
        model = "sora-2-pro"

    size = get_aspect_ratio(context)

    # NOTE sora seems terrible at image to video
    reference_image = None
    if context.image:
        reference_image = convert_pil_to_bytes(resize_image_to_aspect_ratio(context.image, context))

    seconds: Literal["4", "8"] = "8" if context.long_video() else "4"

    try:
        video = client.videos.create_and_poll(
            model=model,
            prompt=context.data.prompt,
            input_reference=reference_image or Omit(),
            size=size,
            seconds=seconds,
            timeout=60 * 10,  # 10 minutes
        )

        if video.status not in ("succeeded", "completed"):
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
