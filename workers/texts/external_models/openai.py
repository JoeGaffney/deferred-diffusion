import copy
from typing import Any, Dict

from openai import OpenAI

from common.logger import logger
from texts.context import TextContext
from texts.schemas import ModelName

OPEN_AI_MODEL_MAP: Dict[ModelName, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4": "gpt-4.1-mini",
    "gpt-5": "gpt-5-mini",
}


def main(context: TextContext) -> str:
    client = OpenAI()
    message: Dict[str, Any] = {
        "role": "user",
        "content": [{"type": "input_text", "text": context.data.prompt}],
    }

    system_message: Dict[str, Any] = {
        "role": "system",
        "content": [{"type": "input_text", "text": context.data.system_prompt}],
    }

    # apply image and video to last message
    for image in context.data.images:
        if message.get("content") is None:
            message["content"] = []

        message["content"].append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image}",
            }
        )

    model = OPEN_AI_MODEL_MAP.get(context.model, "gpt-4o-mini")
    try:
        response = client.responses.create(
            model=model,
            input=[message],  # type: ignore
            instructions=context.data.system_prompt,
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise

    output = response.output_text

    return output
