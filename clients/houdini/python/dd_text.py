import time

import hou
from httpx import RemoteProtocolError

from config import client
from generated.api_client.api.texts import texts_create, texts_get
from generated.api_client.models import (
    SystemPrompt,
    TaskStatus,
    TextCreateResponse,
    TextRequest,
    TextRequestModel,
    TextResponse,
)
from generated.api_client.types import UNSET
from utils import (
    COMPLETED_STATUS,
    get_node_parameters,
    houdini_error_handling,
    input_to_base64,
    polling_message,
    set_node_info,
    threaded,
)


@threaded
def _api_get_call(node, id, iterations=100, sleep_time=5, set_value="response"):
    set_node_info(node, TaskStatus.PENDING, "")

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

            hou.ui.postEventCallback(progress_update)
        except RemoteProtocolError:
            continue  # Retry on protocol errors attempt again
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

            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            node.parm(set_value).set(parsed.result.response)
            set_node_info(node, TaskStatus.SUCCESS, "")

    hou.ui.postEventCallback(update_ui)


def _api_call(node, body: TextRequest):
    try:
        parsed = texts_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, TextCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    node.parm("task_id").set(str(parsed.id))
    _api_get_call(node, str(parsed.id))


def process_text(node):
    set_node_info(node, None, "")
    with houdini_error_handling(node):
        params = get_node_parameters(node)
        images = []
        for current_image in ["a", "b"]:
            image = input_to_base64(node, f"image_{current_image}")
            if image:
                images.append(image)

        model = TextRequestModel(params.get("model", UNSET))
        body = TextRequest(prompt=params.get("prompt", UNSET), images=images, model=model)

        _api_call(node, body)


def get_text(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the text.")

        _api_get_call(node, task_id, iterations=1, sleep_time=0)


def prompt_optimizer(node, prompt: str, system_prompt: SystemPrompt, images: list, model="gpt-5"):
    model = TextRequestModel(model)
    body = TextRequest(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        images=images,
    )

    try:
        parsed = texts_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, TextCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    _api_get_call(node, str(parsed.id), sleep_time=1, iterations=100, set_value="prompt")
