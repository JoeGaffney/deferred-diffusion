import pytest

from tests.videos.helpers import image_to_video, video_to_video, video_upscale


@pytest.mark.parametrize("model", ["runway-gen-4"])
def test_image_to_video(model):
    image_to_video(model)


@pytest.mark.parametrize("model", ["runway-act-two", "runway-aleph"])
def test_video_to_video(model):
    video_to_video(model)


@pytest.mark.parametrize("model", ["runway-upscale"])
def test_video_upscale(model):
    video_upscale(model)
