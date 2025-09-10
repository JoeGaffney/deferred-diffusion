import pytest

from tests.videos.helpers import image_to_video, image_to_video_portrait, text_to_video


@pytest.mark.parametrize("model", ["wan-2-1", "wan-2-2"])
def test_text_to_video(model):
    text_to_video(model)


@pytest.mark.parametrize("model", ["wan-2-2"])
def test_image_to_video(model):
    image_to_video(model)


@pytest.mark.parametrize("model", ["wan-2-2"])
def test_image_to_video_portrait(model):
    image_to_video_portrait(model)
