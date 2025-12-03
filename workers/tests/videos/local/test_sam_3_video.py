import pytest

from tests.videos.helpers import video_segmentation


@pytest.mark.parametrize("model", ["sam-3"])
def test_video_segmentation(model):
    video_segmentation(model)
