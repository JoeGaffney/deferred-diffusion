import pytest

from tests.videos.helpers import (
    first_frame_last_frame,
    image_to_video,
    image_to_video_portrait,
    text_to_video,
    text_to_video_portrait,
)


@pytest.mark.parametrize("model", ["wan-2"])
def test_text_to_video(model):
    text_to_video(model)


@pytest.mark.parametrize("model", ["wan-2"])
def test_text_to_video_portrait(model):
    text_to_video_portrait(model)


@pytest.mark.parametrize("model", ["wan-2"])
def test_image_to_video(model):
    image_to_video(model)


@pytest.mark.parametrize("model", ["wan-2"])
def test_image_to_video_portrait(model):
    image_to_video_portrait(model)


@pytest.mark.parametrize("model", ["wan-2"])
def test_first_frame_last_frame(model):
    first_frame_last_frame(model)
