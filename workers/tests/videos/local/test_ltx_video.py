import pytest

from tests.videos.helpers import image_to_video, text_to_video, video_to_video


@pytest.mark.basic
@pytest.mark.parametrize("model", ["ltx-video"])
def test_text_to_video(model):
    text_to_video(model)


@pytest.mark.parametrize("model", ["ltx-video"])
def test_image_to_video(model):
    image_to_video(model)


@pytest.mark.parametrize("model", ["ltx-video"])
def test_video_to_video(model):
    video_to_video(model)
