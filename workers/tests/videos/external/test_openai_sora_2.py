from typing import List

import pytest

from tests.videos.helpers import (
    image_to_video,
    text_to_video,
    text_to_video_portrait,
    video_to_video,
    video_upscale,
)
from videos.schemas import ModelName

models: List[ModelName] = ["openai-sora-2"]


@pytest.mark.parametrize("model", models)
def test_text_to_video(model):
    text_to_video_portrait(model)
