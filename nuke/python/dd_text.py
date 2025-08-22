import json
import time

import nuke

from config import client
from generated.api_client.api.texts import texts_create, texts_get
from generated.api_client.models.message_content import MessageContent
from generated.api_client.models.message_item import MessageItem
from generated.api_client.models.text_create_response import TextCreateResponse
from generated.api_client.models.text_request import TextRequest
from generated.api_client.models.text_request_model import TextRequestModel
from generated.api_client.models.text_response import TextResponse
from generated.api_client.types import UNSET
from utils import (
    get_node_value,
    get_previous_text_messages,
    node_to_base64,
    nuke_error_handling,
    polling_message,
    set_node_info,
    set_node_value,
    threaded,
)


def get_messages(node):

    # grab the previous messages from the input node
    previous_messages_str = get_previous_text_messages(node.input(0))
    nuke.tprint(f"previous_messages: {previous_messages_str}")

    prompt = get_node_value(node, "prompt", "", mode="get")
    message = [
        MessageItem(
            role="user",
            content=[
                MessageContent(type_="input_text", text=prompt),
            ],
        )
    ]
    previous_messages = []
    try:
        raw_previous = json.loads(previous_messages_str)
        for msg in raw_previous:
            contents = []
            for content_item in msg.get("content", []):
                contents.append(
                    MessageContent(type_=content_item.get("type", "input_text"), text=content_item.get("text", ""))
                )
            previous_messages.append(MessageItem(role=msg.get("role", "user"), content=contents))
    except Exception as e:
        previous_messages = []
    return previous_messages + message


def create_dd_text_node():
    node = nuke.createNode("dd_text")
    return node


@threaded
def _api_get_call(node, id, iterations=1, sleep_time=5):
    set_node_info(node, "PENDING", "")
    for count in range(1, iterations + 1):
        try:
            parsed = texts_get.sync(id, client=client)
            if not isinstance(parsed, TextResponse):
                break
            if parsed.status in ["SUCCESS", "COMPLETED", "ERROR", "FAILED", "FAILURE"]:
                break

            def progress_update():
                if isinstance(parsed, TextResponse):
                    set_node_info(node, parsed.status, polling_message(count, iterations, sleep_time))

            nuke.executeInMainThread(progress_update)
            time.sleep(sleep_time)
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

            chain_of_thought_str = json.dumps(parsed.result.chain_of_thought)
            set_node_value(node, "chain_of_thought", chain_of_thought_str)
            set_node_value(node, "chain_of_thought_alt", chain_of_thought_str)

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

        # Get image inputs a and b as base64
        image_a = node_to_base64(node.input(1), nuke.frame())
        image_b = node_to_base64(node.input(2), nuke.frame())
        images = []
        if image_a:
            images.append(image_a)
        if image_b:
            images.append(image_b)
        body = TextRequest(
            messages=get_messages(node),
            model=model,
            images=images,
        )
        _api_call(node, body)


def get_text(node):
    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the text.")
        _api_get_call(node, task_id, iterations=1, sleep_time=5)
