import json

import hou

from config import MAX_ADDITIONAL_IMAGES, client
from generated.api_client.api.texts import texts_create
from generated.api_client.models.text_request import TextRequest
from generated.api_client.models.text_response import TextResponse
from utils import extract_and_format_parameters, save_tmp_image


def split_text(text, max_length=120):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def get_messages(params):

    # get the current message
    message_content = []

    for i in range(MAX_ADDITIONAL_IMAGES):
        if f"image_{i}_path" in params:
            message_content.append(
                {
                    "type": "image",
                    "image": params[f"image_{i}_path"],
                }
            )
    message_content.append({"type": "text", "text": params.get("prompt", "")})
    message = [
        {
            "role": "user",
            "content": message_content,
        }
    ]

    # factor previous chain of thought
    previous_messages = []
    previous_messages_str = params.get("previous_messages", "[]")
    try:
        previous_messages = json.loads(previous_messages_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        previous_messages = []

    messages = previous_messages + message
    return messages


def main(node):
    # gather our parameters and save any temporary images
    for i in range(MAX_ADDITIONAL_IMAGES):
        save_tmp_image(node, f"tmp_image_{i}")

    params = extract_and_format_parameters(node)
    params["messages"] = get_messages(params)
    valid_params = {k: v for k, v in params.items() if k in TextRequest.__annotations__}
    body = TextRequest(**valid_params)

    # make the API call
    response = texts_create.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, TextResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    # apply back to the node
    chain_of_thought_str = json.dumps(response.parsed.chain_of_thought, indent=2)
    response_str = split_text(str(response.parsed.response))

    node.parm("chain_of_thought").set(chain_of_thought_str)
    node.parm("response").set(response_str)
