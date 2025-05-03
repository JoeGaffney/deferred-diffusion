import base64

from PIL import Image

from agentic.agents.sequence_agent import SequenceRequest, SequenceResponse, main
from tests.utils import image_to_base64
from texts.schemas import TextRequest
from utils.utils import load_image_from_base64

# def test_sequence_agent_image_conversion():
#     """Test the conversion of an image to base64."""

#     image_a = image_to_base64("../test_data/color_v001.jpeg")
#     pil_image = load_image_from_base64(image_a)

#     assert isinstance(pil_image, Image.Image), "Expected 'result' to be a PIL image"


def test_sequence_agent_image_passing():
    sequence_data = SequenceRequest(
        prompt="Create a sequence about an adventure.",
        scene_reference_image=image_to_base64("../test_data/color_v001.jpeg"),
    )
    image_bytes = bytes(sequence_data.scene_reference_image)  # decode to raw bytes

    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sequence_data.prompt},
                ],
            }
        ],
        images=[sequence_data.scene_reference_image] if sequence_data.scene_reference_image else [],
    )
    encoded_image = data.images[0]
    pil_image = load_image_from_base64(encoded_image)

    assert isinstance(pil_image, Image.Image), "Expected 'result' to be a PIL image"


# def test_sequence_agent_with_references():
#     result = main(
#         SequenceRequest(
#             prompt="Create a sequence about an adventure.",
#             scene_reference_image=image_to_base64("../test_data/color_v001.jpeg"),
#             # protagonist_reference_image=image_to_base64("../test_data/face_v001.jpeg"),
#             # antagonist_reference_image=image_to_base64("../test_data/face_v002.jpeg"),
#         )
#     )

#     assert isinstance(result, SequenceResponse), "Expected 'result' to be a SequenceResponse"
