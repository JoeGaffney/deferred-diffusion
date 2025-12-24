from typing import List

import pytest

from tests.videos.helpers import image_to_video, text_to_video
from videos.schemas import ModelName

models: List[ModelName] = ["hailuo-2"]


@pytest.mark.parametrize("model", models)
def test_text_to_video(model):
    text_to_video(model)


@pytest.mark.parametrize("model", models)
def test_image_to_video(model):
    image_to_video(model)
