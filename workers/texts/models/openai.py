import copy

from openai import OpenAI

from common.logger import log_pretty, logger
from texts.context import TextContext


def main(context: TextContext):
    client = OpenAI()
    # messages = context.data.messages
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
    logger.warning(f"Running {context.data.model} with {len(context.data.images)} images")

    response = client.responses.create(
        model=context.data.model_path,
        input=messages,
    )

    output = response.output_text
    print(response.output_text)

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

    log_pretty(f"{context.data.model} result", result)
    return result
