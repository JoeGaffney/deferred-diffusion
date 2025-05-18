import time

import pytest

from tests.utils import image_to_base64
from texts.context import TextContext
from texts.models.qwen_2_5_vl_instruct import main
from texts.schemas import MessageContent, MessageItem, TextRequest

image_a = image_to_base64("../assets/color_v001.jpeg")
image_b = image_to_base64("../assets/style_v001.jpeg")
video_a = image_to_base64("../assets/video_v001.mp4")


def validate_result(result, expected_keyword=None):
    """Helper function to validate main() output structure."""
    assert isinstance(result, dict), "Expected result to be a dictionary"
    assert "response" in result, "Missing 'response' key in result"
    assert "chain_of_thought" in result, "Missing 'chain_of_thought' key in result"

    # Ensure response is valid
    assert isinstance(result["response"], str), "Expected 'response' to be a string"
    assert len(result["response"]) > 0, "Expected non-empty 'response'"

    # Ensure chain_of_thought is valid
    assert isinstance(result["chain_of_thought"], (list)), "Expected 'chain_of_thought' to be a list"

    # If expected_keyword is provided, check if it's in the response
    if expected_keyword:
        assert expected_keyword.lower() in result["response"].lower(), f"Expected '{expected_keyword}' in response"


def test_image_description():
    data = TextRequest(
        messages=[
            MessageItem(role="user", content=[MessageContent(type="text", text="Describe this image.")]),
        ],
        images=[image_a],
    )
    result = main(TextContext(data))
    validate_result(result)


def test_image_prompt_generation():
    data = TextRequest(
        messages=[
            MessageItem(
                role="user",
                content=[
                    MessageContent(
                        type="text", text="Generate a prompt for SD image generation to generate similar images."
                    ),
                ],
            ),
        ],
        images=[image_a],
    )
    result = main(TextContext(data))
    validate_result(result)


def test_image_video_comparison():
    data = TextRequest(
        messages=[
            MessageItem(
                role="user",
                content=[
                    MessageContent(
                        type="text",
                        text="Tell me the differences between the images and videos. I want to know the differences not the content of each.",
                    ),
                ],
            ),
        ],
        images=[image_a, image_b],
    )
    result = main(TextContext(data))
    validate_result(result)
