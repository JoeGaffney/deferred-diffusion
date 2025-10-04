from common.logger import logger
from common.replicate_helpers import process_replicate_video_output, replicate_run
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def main(context: VideoContext):
    payload = {
        "prompt": context.data.prompt,
        "seed": context.data.seed,
        "aspect_ratio": "16:9",
        "resolution": "1080p",
    }

    if context.image:
        payload["first_frame_image"] = convert_pil_to_bytes(context.image)

    if context.image_last_frame:
        payload["last_frame_image"] = convert_pil_to_bytes(context.image_last_frame)

    output = replicate_run("minimax/hailuo-02", payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)
