from PIL import Image

from agents.sequence_agent import SequenceRequest, SequenceResponse, main
from utils.utils import image_to_base64, load_image_from_base64


def test_sequence_agent_image_conversion():
    """Test the conversion of an image to base64."""

    image_a = image_to_base64("../../assets/color_v001.jpeg")
    pil_image = load_image_from_base64(image_a)

    assert isinstance(pil_image, Image.Image), "Expected 'result' to be a PIL image"


def test_sequence_agent_basic():
    result = main(
        SequenceRequest(
            prompt="Create a sequence about an adventure.",
        )
    )

    assert isinstance(result, SequenceResponse), "Expected 'result' to be a SequenceResponse"


def test_sequence_agent_with_references():
    result = main(
        SequenceRequest(
            prompt="Create a sequence about an adventure.",
            scene_reference_image=image_to_base64("../../assets/color_v001.jpeg"),
        )
    )

    assert isinstance(result, SequenceResponse), "Expected 'result' to be a SequenceResponse"
