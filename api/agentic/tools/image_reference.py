from texts.context import TextContext
from texts.models.qwen_2_5_vl_instruct import main as qwen_main
from texts.schemas import TextRequest


def main(prompt, image_reference_image: str) -> str:

    data = TextRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        images=[image_reference_image],
    )
    result = qwen_main(TextContext(data))
    return result["response"]
