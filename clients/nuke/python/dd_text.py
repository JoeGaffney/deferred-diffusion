import time

import nuke
from httpx import RemoteProtocolError

from config import client
from generated.api_client.api.texts import texts_create, texts_get
from generated.api_client.models import (
    TextCreateResponse,
    TextRequest,
    TextRequestModel,
    TextResponse,
)
from generated.api_client.types import UNSET
from utils import (
    COMPLETED_STATUS,
    get_node_value,
    node_to_base64,
    nuke_error_handling,
    polling_message,
    set_node_info,
    set_node_value,
    threaded,
)


def create_dd_text_node():
    node = nuke.createNode("dd_text")
    return node


@threaded
def _api_get_call(node, id, iterations=1, sleep_time=5):
    set_node_info(node, "PENDING", "")
    for count in range(1, iterations + 1):
        time.sleep(sleep_time)

        try:
            parsed = texts_get.sync(id, client=client)
            if not isinstance(parsed, TextResponse):
                break

            if parsed.status in COMPLETED_STATUS:
                break

            def progress_update(parsed=parsed, count=count):
                set_node_info(node, parsed.status, polling_message(count, iterations, sleep_time))

            nuke.executeInMainThread(progress_update)
        except RemoteProtocolError:
            continue  # Retry on protocol errors attempt again
        except Exception as e:

            def handle_error(error=e):
                with nuke_error_handling(node):
                    raise RuntimeError(f"API call failed: {str(error)}") from error

            nuke.executeInMainThread(handle_error)
            return

    def update_ui():
        with nuke_error_handling(node):
            if not isinstance(parsed, TextResponse):
                raise ValueError("Unexpected response type from API call.")
            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            response_str = str(parsed.result.response)
            set_node_value(node, "response", response_str)
            set_node_info(node, "COMPLETE", "")

    nuke.executeInMainThread(update_ui)


def _api_call(node, body: TextRequest):
    try:
        parsed = texts_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, TextCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    set_node_value(node, "task_id", str(parsed.id))
    _api_get_call(node, str(parsed.id), iterations=100)


def process_text(node):
    set_node_info(node, "", "")

    with nuke_error_handling(node):
        model = TextRequestModel(get_node_value(node, "model", UNSET, mode="value"))
        prompt = get_node_value(node, "prompt", "", mode="get")

        # Get image inputs a and b as base64
        image_a = node_to_base64(node.input(0), nuke.frame())
        image_b = node_to_base64(node.input(1), nuke.frame())
        images = []
        if image_a:
            images.append(image_a)
        if image_b:
            images.append(image_b)

        body = TextRequest(
            model=model,
            prompt=prompt,
            images=images,
        )
        _api_call(node, body)


def get_text(node):
    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the text.")
        _api_get_call(node, task_id, iterations=1, sleep_time=5)
