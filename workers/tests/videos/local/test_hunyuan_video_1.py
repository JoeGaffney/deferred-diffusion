from typing import List

import pytest

from tests.videos.helpers import (
    ModelName,
    image_to_video,
    image_to_video_portrait,
    text_to_video,
    text_to_video_portrait,
)

models: List[ModelName] = ["hunyuan-video-1"]


@pytest.mark.parametrize("model", models)
def test_text_to_video(model):
    text_to_video(model, num_frames=121)


@pytest.mark.parametrize("model", models)
def test_text_to_video_portrait(model):
    text_to_video_portrait(model)


@pytest.mark.parametrize("model", models)
def test_image_to_video(model):
    image_to_video(model)


@pytest.mark.parametrize("model", models)
def test_image_to_video_portrait(model):
    image_to_video_portrait(model)
