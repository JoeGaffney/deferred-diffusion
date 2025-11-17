from typing import List

import pytest

from tests.videos.helpers import video_upscale
from videos.schemas import ModelName

models: List[ModelName] = ["runway-upscale"]


@pytest.mark.parametrize("model", models)
def test_video_upscale(model):
    video_upscale(model)
