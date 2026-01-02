from pathlib import Path
from typing import List

from common.replicate_helpers import process_replicate_video_output, replicate_run
from videos.context import VideoContext


def main(context: VideoContext) -> List[Path]:
    if context.data.video is None:
        raise ValueError("Input video is None. Please provide a valid video.")

    model = "runwayml/upscale-v1"
    video_uri = f"data:video/mp4;base64,{context.data.video}"
    payload = {
        "video": video_uri,
    }

    output = replicate_run(model, payload)
    video_url = process_replicate_video_output(output)

    return [context.save_output_url(video_url)]
