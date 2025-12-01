import time

import hou
from httpx import RemoteProtocolError

from config import client
from dd_text import prompt_optimizer
from generated.api_client.api.images import images_create, images_get
from generated.api_client.models import (
    ImageCreateResponse,
    ImageRequest,
    ImageRequestModel,
    ImageResponse,
    SystemPrompt,
    TaskStatus,
)
from generated.api_client.types import UNSET
from utils import (
    COMPLETED_STATUS,
    base64_to_image,
    get_node_parameters,
    get_output_path,
    get_references,
    houdini_error_handling,
    input_to_base64,
    polling_message,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def _api_get_call(node, id, output_path: str, expanded_path: str, iterations=100, sleep_time=5):
    set_node_info(node, TaskStatus.PENDING, "")

    for count in range(1, iterations + 1):
        time.sleep(sleep_time)

        try:
            parsed = images_get.sync(id, client=client)
            if not isinstance(parsed, ImageResponse):
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
            if not isinstance(parsed, ImageResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the image to the specified path before reloading the outputs
            base64_to_image(parsed.result.base64_data, expanded_path, save_copy=True)

            node.parm("output_image_path").set(output_path)
            reload_outputs(node, "output_read")
            set_node_info(node, parsed.status, output_path)

    hou.ui.postEventCallback(update_ui)


def _api_call(node, body: ImageRequest, output_image_path: str):
    try:
        parsed = images_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, ImageCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    node.parm("task_id").set(str(parsed.id))
    _api_get_call(node, str(parsed.id), output_image_path, hou.expandString(output_image_path))


def process_image(node):
    set_node_info(node, None, "")
    with houdini_error_handling(node):
        params = get_node_parameters(node)
        output_image_path = get_output_path(node, movie=False)
        image = input_to_base64(node, "src")
        mask = input_to_base64(node, "mask")

        body = ImageRequest(
            model=ImageRequestModel(params.get("model", "sd-xl")),
            image=image,
            mask=mask,
            height=params.get("height", UNSET),
            width=params.get("width", UNSET),
            prompt=params.get("prompt", UNSET),
            seed=params.get("seed", UNSET),
            strength=params.get("strength", UNSET),
            references=get_references(node),
        )

        _api_call(node, body, output_image_path)


def get_image(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the image.")

        output_image_path = get_output_path(node, movie=False)
        _api_get_call(
            node, task_id, output_image_path, hou.expandString(output_image_path), iterations=1, sleep_time=0
        )


def image_prompt_optimizer(node):
    params = get_node_parameters(node)
    text_model = params.get("text_model", "gpt-5")
    prompt = params.get("prompt", "")
    image = input_to_base64(node, "src")

    images = []
    if image:
        images.append(image)

    prompt_optimizer(node, prompt, SystemPrompt.IMAGE_OPTIMIZER, images, model=text_model)
