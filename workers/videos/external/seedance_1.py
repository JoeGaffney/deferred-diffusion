from pathlib import Path
from typing import List

from common.replicate_helpers import process_replicate_video_output, replicate_run
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def get_aspect_ratio(context: VideoContext) -> str:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "16:9"
    elif dimension_type == "portrait":
        return "9:16"
    return "1:1"


def get_resolution(context: VideoContext) -> str:
    if context.get_mega_pixels() >= 2.0:  # close to 1080p or higher
        return "1080p"
    return "720p"


def main(context: VideoContext) -> List[Path]:
    model = "bytedance/seedance-1.5-pro"
    payload = {
        "prompt": context.data.cleaned_prompt,
        "seed": context.data.seed,
        "aspect_ratio": get_aspect_ratio(context),
        "duration": 5 if context.duration_in_seconds() <= 5 else 10,
        # "resolution": get_resolution(context),# Currently not used by the model
        "generate_audio": context.data.generate_audio,
    }

    if context.image:
        payload["image"] = convert_pil_to_bytes(context.image)

    if context.last_image:
        payload["last_frame_image"] = convert_pil_to_bytes(context.last_image)

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return [context.save_output_url(video_url)]
