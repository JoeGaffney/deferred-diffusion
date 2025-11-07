import pytest
from pydantic import ValidationError

from images.schemas import ImageRequest, References


def test_mask_requires_image():
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="sd-xl", prompt="p", mask="base64mask")
    assert "mask requires image" in str(e.value)


def test_gemini_rejects_inpainting():
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="google-gemini-2", prompt="p", image="base64img", mask="base64mask")
    assert "does not support mode 'image_to_image_inpainting'" in str(e.value)


def test_references_not_supported_for_depth():
    refs = [References(mode="style", strength=0.5, image="base64img")]
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="depth-anything-2", prompt="p", image="base64img", references=refs)
    assert "does not support references" in str(e.value)


def test_depth_model_rejects_text_only():
    with pytest.raises(ValidationError) as e:
        ImageRequest(model="depth-anything-2", prompt="p")
    assert "does not support mode 'text_to_image'" in str(e.value)
