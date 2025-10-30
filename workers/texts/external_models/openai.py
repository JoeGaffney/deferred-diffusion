import copy
from typing import Dict

from openai import OpenAI

from common.logger import log_pretty, logger
from texts.context import TextContext
from texts.schemas import ModelName

OPEN_AI_MODEL_MAP: Dict[ModelName, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4": "gpt-4.1-mini",
    "gpt-5": "gpt-5-mini",
}


def main(context: TextContext):
    client = OpenAI()
    messages = [message.model_dump() for message in context.data.messages]
    original_messages = copy.deepcopy(messages)

    # apply image and video to last message
    last_message = messages[-1]
    for image in context.data.images:
        last_message_content = last_message.get("content", [])
        last_message_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image}",
            }
        )

    model = OPEN_AI_MODEL_MAP.get(context.model, "gpt-4o-mini")
    response = client.responses.create(
        model=model,
        input=messages,  # type: ignore
    )

    output = response.output_text

    # we keep only the original as we may have altered adding video and image to the last message
    chain_of_thought = original_messages
    chain_of_thought.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": output,
                }
            ],
        }
    )
    result = {
        "response": output,
        "chain_of_thought": chain_of_thought,
    }

    return result
