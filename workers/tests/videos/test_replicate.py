import pytest

from tests.videos.helpers import (
    image_to_video,
    text_to_video,
    video_to_video,
    video_upscale,
)


@pytest.mark.parametrize("model", ["bytedance-seedance-1", "minimax-hailuo-2", "google-veo-3"])
def test_text_to_video(model):
    text_to_video(model)


@pytest.mark.parametrize("model", ["bytedance-seedance-1", "kwaivgi-kling-2", "minimax-hailuo-2"])
def test_image_to_video(model):
    image_to_video(model)
