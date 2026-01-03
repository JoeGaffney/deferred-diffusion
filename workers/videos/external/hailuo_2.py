from pathlib import Path
from typing import List

from common.replicate_helpers import process_replicate_video_output, replicate_run
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def get_resolution(context: VideoContext) -> str:
    if context.get_mega_pixels() >= 2.0:  # close to 1080p or higher
        return "1080p"
    return "768p"


def main(context: VideoContext) -> List[Path]:
    model = "minimax/hailuo-2.3"
    payload = {
        "prompt": context.data.cleaned_prompt,
        "duration": 6 if context.duration_in_seconds() <= 5 else 10,
        "resolution": get_resolution(context),
        "prompt_optimizer": False,
    }

    # Add first frame image if available
    if context.image:
        payload["first_frame_image"] = convert_pil_to_bytes(context.image)  # type: ignore[assignment]

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return [context.save_output_url(video_url)]
