import os
import shutil
from typing import List

import pytest

from tests.utils import image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest
from videos.tasks import model_router_main as main

MODES = ["text_to_video"]
models: List[ModelName] = ["wan-2-2"]


@pytest.mark.parametrize("mode", ["text_to_video"])
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("seed", [1])
def test_text_to_video(model, mode, seed):
    output_name = setup_output_file(model, mode, suffix=f"_{seed}", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                prompt="A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature.",
                num_inference_steps=10,
                guidance_scale=3.0,
                num_frames=24,
                seed=seed,
            )
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
