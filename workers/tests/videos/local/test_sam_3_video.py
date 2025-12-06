import pytest

from tests.videos.helpers import video_segmentation, video_segmentation_alt


@pytest.mark.parametrize("model", ["sam-3"])
def test_video_segmentation(model):
    video_segmentation(model)


@pytest.mark.parametrize("model", ["sam-3"])
def test_video_segmentation_alt(model):
    video_segmentation_alt(model)
