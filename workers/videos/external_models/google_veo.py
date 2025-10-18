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
    return "16:9"


def main(context: VideoContext):

    model = "google/veo-3.1-fast"
    if context.data.high_quality:
        model = "google/veo-3.1"

    duration = 8 if context.long_video() else 4

    # cheaper without audio
    generate_audio = False

    payload = {
        "prompt": context.data.prompt,
        "seed": context.data.seed,
        "aspect_ratio": get_aspect_ratio(context),
        "duration": duration,
        "generate_audio": generate_audio,
    }

    if context.image:
        payload["image"] = convert_pil_to_bytes(context.image)

        if context.image_last_frame:
            payload["last_frame"] = convert_pil_to_bytes(context.image_last_frame)

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)
