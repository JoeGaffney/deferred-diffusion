import os
import shutil
from typing import List

import pytest

from tests.utils import image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest
from videos.tasks import external_model_router_main as main

MODES = ["video_upscale"]
models: List[ModelName] = ["external-runway-upscale"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", models)
def test_image_to_video(model, mode):
    output_name = setup_output_file(model, mode, extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                video=image_to_base64("../assets/act_reference_v001.mp4"),
                num_frames=24,
                num_inference_steps=7,
                guidance_scale=3.0,
            )
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
