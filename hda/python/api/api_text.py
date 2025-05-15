import json

import hou

from config import MAX_ADDITIONAL_IMAGES, client
from generated.api_client.api.texts import texts_create, texts_get
from generated.api_client.models.text_create_response import TextCreateResponse
from generated.api_client.models.text_request import TextRequest
from generated.api_client.models.text_request_model import TextRequestModel
from generated.api_client.models.text_response import TextResponse
from generated.api_client.types import UNSET
from utils import (
    ApiResponseError,
    get_node_parameters,
    handle_api_response,
    input_to_base64,
    set_node_info,
    threaded,
)


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
    message = [
        {
            "role": "user",
            "content": [{"type": "text", "text": params.get("prompt", "")}],
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


@threaded
def api_get_call(id, node):
    try:
        response = texts_get.sync_detailed(id, client=client)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        hou.ui.displayMessage(str(e))
        return

    def update_ui():
        try:
            parsed = handle_api_response(response, TextResponse, "API Get Call Failed")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ApiResponseError(
                    f"Task {parsed.status} with error: {parsed.error_message}", status=parsed.status
                )
        except ApiResponseError as e:
            set_node_info(node, e.status, str(e))
            hou.ui.displayMessage(str(e))
            return

        # apply back to the node
        chain_of_thought_str = json.dumps(parsed.result.chain_of_thought, indent=2)
        response_str = split_text(str(parsed.result.response))

        node.parm("chain_of_thought").set(chain_of_thought_str)
        node.parm("response").set(response_str)
        set_node_info(node, "COMPLETE", "")

    hou.ui.postEventCallback(update_ui)


def main(node):
    set_node_info(node, "", "")

    params = get_node_parameters(node)
    images = []
    for current_image in range(MAX_ADDITIONAL_IMAGES):
        image = input_to_base64(node, f"image_{current_image}")
        if image:
            images.append(image)

    model = TextRequestModel(params.get("model", UNSET))
    body = TextRequest(messages=get_messages(params), images=images, model=model)
    print(f"TextRequest: {body.messages}")
    # make the initial API call to create the text task
    id = None
    try:
        response = texts_create.sync_detailed(client=client, body=body)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        raise

    try:
        parsed = handle_api_response(response, TextCreateResponse)

        id = parsed.id
        if not id:
            raise ApiResponseError("No ID found in the response.")
    except ApiResponseError as e:
        set_node_info(node, e.status, str(e))
        raise

    set_node_info(node, "PENDING", "", str(id))
    api_get_call(id, node)
