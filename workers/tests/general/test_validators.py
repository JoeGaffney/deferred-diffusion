import pytest
from pydantic import ValidationError

from images.schemas import ImageRequest, References
from videos.schemas import VideoRequest


def test_images_validation():
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="sd-xl", prompt="p", mask="base64mask")
    assert "mask requires image" in str(e.value)

    with pytest.raises(ValidationError) as e:
        ImageRequest(model="google-gemini-2", prompt="p", image="base64img", mask="base64mask")
    assert "does not support mode 'inpainting'" in str(e.value)

    refs = [References(mode="style", strength=0.5, image="base64img")]
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="depth-anything-2", prompt="p", image="base64img", references=refs)
    assert "does not support references" in str(e.value)

    with pytest.raises(ValidationError) as e:
        ImageRequest(model="depth-anything-2", prompt="p")
    assert "does not support mode 'text_to_image'" in str(e.value)


def test_videos_validation():
    # Test last_image requires image
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="ltx-video", prompt="p", last_image="base64lastimg")
    assert "last_image requires image" in str(e.value)

    # Test video requires image
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="ltx-video", prompt="p", video="base64video")
    assert "video requires image" in str(e.value)

    # Test model that doesn't support text_to_video mode
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="runway-upscale", prompt="p")
    assert "does not support mode 'text_to_video'" in str(e.value)

    # Test model that doesn't support image_to_video mode
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="minimax-hailuo-2", prompt="p", image="base64img")
    assert "does not support mode 'image_to_video'" in str(e.value)

    # Test model that doesn't support video_to_video mode
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="bytedance-seedance-1", prompt="p", image="base64img", video="base64video")
    assert "does not support mode 'video_to_video'" in str(e.value)

    # Test model that doesn't support first_last_frame mode
    with pytest.raises(ValidationError) as e:
        VideoRequest(model="runway-gen-4", prompt="p", image="base64img", last_image="base64lastimg")
    assert "does not support mode 'first_last_frame'" in str(e.value)
