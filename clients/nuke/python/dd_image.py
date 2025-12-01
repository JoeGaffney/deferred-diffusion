import time

import nuke
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
    base64_to_file,
    get_node_value,
    get_output_path,
    get_references,
    node_to_base64,
    nuke_error_handling,
    polling_message,
    replace_hashes_with_frame,
    set_node_info,
    set_node_value,
    threaded,
    update_read_range,
)


def create_dd_image_node():
    node = nuke.createNode("dd_image")
    return node


@threaded
def _api_get_call(node, id, output_path: str, current_frame: int, iterations=100, sleep_time=5):
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
            if not isinstance(parsed, ImageResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the image to the specified path
            resolved_output_path = replace_hashes_with_frame(output_path, current_frame)
            base64_to_file(parsed.result.base64_data, resolved_output_path)

            output_read = nuke.toNode(f"{node.name()}.output_read")
            set_node_value(output_read, "file", output_path)
            update_read_range(output_read)

            set_node_info(node, TaskStatus.SUCCESS, "")

    nuke.executeInMainThread(update_ui)


def _api_call(node, body: ImageRequest, output_image_path: str, current_frame: int):
    try:
        parsed = images_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, ImageCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    set_node_value(node, "task_id", str(parsed.id))
    _api_get_call(node, str(parsed.id), output_image_path, current_frame)


def process_image(node):
    set_node_info(node, None, "")
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        output_image_path = get_output_path(node, movie=False)
        if not output_image_path:
            raise ValueError("Output image path is required.")

        image_node = node.input(0)
        mask_node = node.input(1)
        image = node_to_base64(image_node, current_frame)
        mask = node_to_base64(mask_node, current_frame)
        width_height = get_node_value(node, "width_height", [1280, 720], return_type=list, mode="value")

        body = ImageRequest(
            model=ImageRequestModel(get_node_value(node, "model", "sd-xl", mode="value")),
            image=image,
            mask=mask,
            prompt=get_node_value(node, "prompt", UNSET, mode="get"),
            seed=get_node_value(node, "seed", UNSET, return_type=int, mode="value"),
            strength=get_node_value(node, "strength", UNSET, return_type=float, mode="value"),
            width=int(width_height[0]),
            height=int(width_height[1]),
            references=get_references(node),
        )
        _api_call(node, body, output_image_path, current_frame)


def get_image(node):
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the image.")

        output_image_path = get_output_path(node, movie=False)
        _api_get_call(node, task_id, output_image_path, current_frame, iterations=1, sleep_time=0)


def image_prompt_optimizer(node):
    current_frame = nuke.frame()
    model = get_node_value(node, "model", "sd-xl", mode="value")
    prompt = get_node_value(node, "prompt", "", mode="get")
    image_node = node.input(0)
    image = node_to_base64(image_node, current_frame)

    images = []
    if image:
        images.append(image)

    prompt_optimizer(node, prompt, SystemPrompt.IMAGE_OPTIMIZER, images)
