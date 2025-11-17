from typing import List

import pytest

from tests.videos.helpers import text_to_video
from videos.schemas import ModelName

models: List[ModelName] = ["google-veo-3"]


@pytest.mark.parametrize("model", models)
def test_text_to_video(model):
    text_to_video(model)
