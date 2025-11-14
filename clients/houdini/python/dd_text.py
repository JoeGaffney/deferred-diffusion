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
    get_node_parameters,
    houdini_error_handling,
    input_to_base64,
    set_node_info,
    threaded,
)


@threaded
def api_get_call(id, node):
    try:
        parsed = texts_get.sync(id, client=client)
    except Exception as e:

        def handle_error(error=e):
            with houdini_error_handling(node):
                raise RuntimeError(f"API call failed: {str(error)}") from error

        hou.ui.postEventCallback(handle_error)
        return

    def update_ui():
        with houdini_error_handling(node):
            if not isinstance(parsed, TextResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            node.parm("response").set(parsed.result.response)
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
    body = TextRequest(prompt=params.get("prompt", UNSET), images=images, model=model)

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

    set_node_info(node, "PENDING", "")
    api_get_call(id, node)
