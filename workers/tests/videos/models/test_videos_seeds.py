import os
import shutil
from typing import List

import pytest

from tests.utils import setup_output_file
from tests.videos.helpers import main
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest

MODES = ["text_to_video"]
models: List[ModelName] = ["wan-2"]


@pytest.mark.parametrize("mode", ["text_to_video"])
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("seed", range(1, 3))
def test_text_to_video(model, mode, seed):
    output_name = setup_output_file(model, mode, suffix=f"_{seed}", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
                num_frames=48,
                seed=seed,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
