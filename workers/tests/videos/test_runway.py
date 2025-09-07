import os
import shutil
from typing import List

import pytest

from tests.utils import image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest
from videos.tasks import model_router_main as main

MODES = ["image_to_image"]
models: List[ModelName] = ["runway-gen-4"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", models)
def test_image_to_video(model, mode):
    output_name = setup_output_file(model, mode, extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/color_v002.png"),
                prompt="A man with short gray hair plays a red electric guitar.",
                num_inference_steps=7,
                guidance_scale=3.0,
                num_frames=24,
            )
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
