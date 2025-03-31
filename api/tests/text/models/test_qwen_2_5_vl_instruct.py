import time

import pytest
from text.context import TextContext
from text.models.qwen_2_5_vl_instruct import main
from text.schemas import TextRequest


@pytest.fixture
def input_paths():
    return {"image": "../test_data/color_v001.jpeg", "video": "../test_data/video_v001.mp4"}


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


def test_image_description(input_paths):
    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_paths["image"]},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
    )
    result = main(TextContext(data), flush_gpu_memory=True)
    validate_result(result)


def test_image_prompt_generation(input_paths):
    time.sleep(2)  # Simulating GPU memory flush delay
    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_paths["image"]},
                    {"type": "text", "text": "Give me a prompt for SD image generation to generate similar images."},
                ],
            }
        ]
    )
    result = main(TextContext(data), flush_gpu_memory=False)
    validate_result(result)


def test_image_video_comparison(input_paths):
    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_paths["image"]},
                    {"type": "video", "video": input_paths["video"]},
                    {
                        "type": "text",
                        "text": "Tell me the differences between the image and video. I want to know the differences not the content of each.",
                    },
                ],
            }
        ]
    )
    result = main(TextContext(data), flush_gpu_memory=False)
    validate_result(result)
