import pytest

from tests.videos.helpers import (
    image_to_video,
    text_to_video,
    text_to_video_portrait,
    video_to_video,
    video_upscale,
)


@pytest.mark.parametrize("model", ["openai-sora-2"])
def test_text_to_video(model):
    text_to_video_portrait(model)


# @pytest.mark.parametrize("model", ["openai-sora-2"])
# def test_image_to_video(model):
#     image_to_video(model)
