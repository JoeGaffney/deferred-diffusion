import copy
from typing import Any, Dict

from openai import OpenAI

from common.logger import logger
from texts.context import TextContext


def main(context: TextContext, model_path="gpt-4o-mini") -> str:
    client = OpenAI()
    message: Dict[str, Any] = {
        "role": "user",
        "content": [{"type": "input_text", "text": context.data.prompt}],
    }

    # NOTE: system message is passed via 'instructions' parameter
    # system_message: Dict[str, Any] = {
    #     "role": "system",
    #     "content": [{"type": "input_text", "text": context.data.system_prompt}],
    # }

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

    try:
        response = client.responses.create(
            model=model_path,
            input=[message],  # type: ignore
            instructions=context.data.system_prompt,
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise

    output = response.output_text

    return output


def main_gpt_4o(context: TextContext) -> str:
    return main(context, model_path="gpt-4o-mini")


def main_gpt_4(context: TextContext) -> str:
    return main(context, model_path="gpt-4.1-mini")


def main_gpt_5(context: TextContext) -> str:
    return main(context, model_path="gpt-5-mini")
