import os
import shutil

from tests.utils import image_to_base64
from utils.utils import ensure_path_exists
from videos.context import VideoContext
from videos.models.hunyuan_video import main
from videos.schemas import VideoRequest


def test_image_to_video():
    output_name = f"../tmp/output/videos/HunyuanVideo.mp4"
    ensure_path_exists(output_name)

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        VideoContext(
            VideoRequest(
                model="HunyuanVideo",
                image=image_to_base64("../assets/color_v002.png"),
                prompt="A man with short gray hair plays a red electric guitar.",
                num_inference_steps=10,
                guidance_scale=1,
                num_frames=12,
            )
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
