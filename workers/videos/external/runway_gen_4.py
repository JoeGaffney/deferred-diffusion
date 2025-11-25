from typing import Literal

from common.replicate_helpers import process_replicate_video_output, replicate_run
from utils.utils import convert_pil_to_bytes
from videos.context import VideoContext


def get_aspect_ratio(context: VideoContext) -> Literal["16:9", "9:16", "1:1"]:
    dimension_type = context.get_dimension_type()
    if dimension_type == "landscape":
        return "16:9"
    elif dimension_type == "portrait":
        return "9:16"
    return "1:1"


def video_to_video(context: VideoContext):
    if context.data.video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    model = "runwayml/gen4-aleph"
    video_uri = f"data:video/mp4;base64,{context.get_compressed_video()}"
    payload = {
        "prompt": context.data.cleaned_prompt,
        "seed": context.data.seed,
        "aspect_ratio": get_aspect_ratio(context),
        "video": video_uri,
    }

    if context.image:
        payload["reference_image"] = convert_pil_to_bytes(context.image)

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)


def image_to_video(context: VideoContext):
    if context.image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    model = "runwayml/gen4-turbo"
    payload = {
        "prompt": context.data.cleaned_prompt,
        "seed": context.data.seed,
        "aspect_ratio": get_aspect_ratio(context),
        "duration": 5 if context.duration_in_seconds() <= 5 else 10,
        "image": convert_pil_to_bytes(context.image),
    }

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return context.save_video_url(video_url)


def main(context: VideoContext):
    if context.data.video:
        return video_to_video(context)
    return image_to_video(context)
