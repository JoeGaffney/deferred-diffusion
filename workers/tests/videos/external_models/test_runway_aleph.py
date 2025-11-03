from typing import List

import pytest

from tests.videos.helpers import video_to_video
from videos.schemas import ModelName

models: List[ModelName] = ["runway-aleph"]


@pytest.mark.parametrize("model", models)
def test_video_to_video(model):
    video_to_video(model)
