import pytest

from tests.utils import image_to_base64
from texts.context import TextContext
from texts.local.qwen_2 import main
from texts.schemas import TextRequest

image_a = image_to_base64("../assets/color_v001.jpeg")
image_b = image_to_base64("../assets/style_v001.jpeg")
video_a = image_to_base64("../assets/video_v001.mp4")


def validate_result(result: str, expected_keyword=None):
    """Helper function to validate main() output structure."""
    # Ensure response is valid
    assert isinstance(result, str), "Expected 'response' to be a string"
    assert len(result) > 0, "Expected non-empty 'response'"
    print("Response:", result)
    # If expected_keyword is provided, check if it's in the response
    if expected_keyword:
        assert expected_keyword.lower() in result.lower(), f"Expected '{expected_keyword}' in response"


@pytest.mark.basic
def test_image_description():
    data = TextRequest(
        model="qwen-2",
        prompt="Describe this image.",
        images=[image_a],
    )
    result = main(TextContext(data))
    validate_result(result)


def test_image_prompt_generation():
    data = TextRequest(
        model="qwen-2",
        prompt="Generate a prompt for SD image generation to generate similar images.",
        images=[image_a],
    )
    result = main(TextContext(data))
    validate_result(result)


def test_image_video_comparison():
    data = TextRequest(
        model="qwen-2",
        prompt="Tell me the differences between the images and videos. I want to know the differences not the content of each.",
        images=[image_b],
        videos=[video_a],
    )
    result = main(TextContext(data))
    validate_result(result)
