import pytest

from tests.videos.helpers import video_to_video


@pytest.mark.parametrize("model", ["wan-2"])
def test_video_to_video(model):
    video_to_video(model)
