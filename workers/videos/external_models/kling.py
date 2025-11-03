from common.logger import logger
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


def main(context: VideoContext):
    if context.image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    model = "kwaivgi/kling-v2.5-turbo-pro"
    payload = {
        "prompt": context.data.prompt,
        "seed": context.data.seed,
        "aspect_ratio": get_aspect_ratio(context),
        "mode": "standard",
    }

    if context.image:
        # 2.5 does not support last frame input
        if context.last_image:
            payload["start_image"] = convert_pil_to_bytes(context.image)
            payload["end_image"] = convert_pil_to_bytes(context.last_image)
            payload["mode"] = "pro"
            model = "kwaivgi/kling-v2.1"
        else:
            payload["image"] = convert_pil_to_bytes(context.image)

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)
