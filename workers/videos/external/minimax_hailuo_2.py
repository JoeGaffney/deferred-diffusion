from common.logger import logger
from common.replicate_helpers import process_replicate_video_output, replicate_run
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def main(context: VideoContext):
    model = "minimax/hailuo-2.3"
    payload = {
        "prompt": context.data.cleaned_prompt,
        "duration": 5 if context.duration_in_seconds() <= 5 else 10,
        "resolution": "1080p" if context.get_resolution_type() == "1080p" else "768p",
        "prompt_optimizer": False,
    }

    # Add first frame image if available
    if context.image:
        payload["first_frame_image"] = convert_pil_to_bytes(context.image)

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)
