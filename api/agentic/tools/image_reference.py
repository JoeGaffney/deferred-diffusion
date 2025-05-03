import os

from pydantic import Base64Bytes

from texts.context import TextContext
from texts.models.qwen_2_5_vl_instruct import main as qwen_main
from texts.schemas import TextRequest


def main(prompt, image_reference_image: Base64Bytes) -> str:

    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_reference_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        images=[image_reference_image],
    )
    result = qwen_main(TextContext(data))
    return result["response"]
