import os
import traceback
from contextlib import contextmanager

import nuke

from config import client
from generated.api_client.api.images import images_create, images_get
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET
from utils import (
    base64_to_image,
    get_control_nets,
    get_ip_adapters,
    get_node_value,
    node_to_base64,
    set_node_value,
    threaded,
)


def create_dd_image_node():
    # Create the node from the gizmo (no need to re-define defaults)
    node = nuke.createNode("dd_image")  # 'dd_image' is the name of your gizmo

    # Optionally: You can set other properties or interact with the node here
    # e.g., If you want to call a function defined inside the gizmo, you can do it here
    return node


def set_node_info(node, status, message):
    # Update the node label to show current status
    status_text = f"[{status}]"
    node["label"].setValue(status_text)

    if status == "COMPLETE":
        node["tile_color"].setValue(0x00CC00FF)  # Green
    elif status == "PENDING":
        node["tile_color"].setValue(0xCCCC00FF)  # Yellow
    elif status == "FAILED" or status == "ERROR" or status == "FAILURE":
        node["tile_color"].setValue(0xCC0000FF)  # Red
        if message:
            node["label"].setValue(f"{status_text} {message}")
    else:
        node["tile_color"].setValue(0x888888FF)  # Grey


@contextmanager
def nuke_error_handling(node):
    try:
        yield
    except ValueError as e:
        set_node_info(node, "ERROR", str(e))
        nuke.message(str(e))
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        traceback.print_exc()
        nuke.message(str(e))


def get_output_path(node, movie=False) -> str:
    node_name = node.name()
    time_stamp = str(node.__hash__())
    # time_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    extension = "#####.png"
    if movie:
        extension = "mp4"

    script_dir = os.path.dirname(nuke.root().name())
    output_image_path = f"{script_dir}/deferred-diffusion/{node_name}/{time_stamp}.{extension}"
    return output_image_path


def replace_hashes_with_frame(path_with_hashes, frame):
    num_hashes = path_with_hashes.count("#")
    if num_hashes == 0:
        # No hashes to replace, return original path
        return path_with_hashes
    frame_str = str(frame).zfill(num_hashes)
    return path_with_hashes.replace("#" * num_hashes, frame_str)


@threaded
def _api_get_call(node, id, output_path: str, wait=False):
    set_node_info(node, "PENDING", "")

    try:
        parsed = images_get.sync(id, client=client, wait=wait)
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

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the image to the specified path
            resolved_output_path = replace_hashes_with_frame(output_path, nuke.frame())
            base64_to_image(parsed.result.base64_data, resolved_output_path)

            output_read = nuke.toNode(f"{node.name()}.output_read")
            set_node_value(output_read, "file", output_path)
            output_read["reload"].execute()

            set_node_info(node, "COMPLETE", output_path)

    nuke.executeInMainThread(update_ui)


def _api_call(node, body: ImageRequest, output_image_path: str):
    try:
        parsed = images_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, ImageCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    set_node_value(node, "task_id", str(parsed.id))
    _api_get_call(node, str(parsed.id), output_image_path, wait=True)


def process_image(node):
    """Process the node using the API"""
    set_node_info(node, "", "")

    with nuke_error_handling(node):
        output_image_path = get_output_path(node, movie=False)
        if not output_image_path:
            raise ValueError("Output image path is required.")

        current_frame = nuke.frame()
        image_node = node.input(0)
        mask_node = node.input(1)
        aux_node = node.input(2)

        image = node_to_base64(image_node, current_frame)
        mask = node_to_base64(mask_node, current_frame)
        width_height = get_node_value(node, "width_height", [1024, 1024], return_type=list, mode="value")

        body = ImageRequest(
            model=ImageRequestModel(get_node_value(node, "model", "sd-xl", mode="value")),
            guidance_scale=get_node_value(node, "guidance_scale", UNSET, return_type=float, mode="value"),
            image=image,
            mask=mask,
            negative_prompt=get_node_value(node, "negative_prompt", UNSET, mode="get"),
            num_inference_steps=get_node_value(node, "num_inference_steps", UNSET, return_type=int, mode="value"),
            prompt=get_node_value(node, "prompt", UNSET, mode="get"),
            seed=get_node_value(node, "seed", UNSET, return_type=int, mode="value"),
            strength=get_node_value(node, "strength", UNSET, return_type=float, mode="value"),
            width=int(width_height[0]),
            height=int(width_height[1]),
            controlnets=get_control_nets(aux_node),
            ip_adapters=get_ip_adapters(aux_node),
        )

        _api_call(node, body, output_image_path)


def get_image(node):
    """Get an image using a task ID"""
    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the image.")

        output_image_path = get_output_path(node, movie=False)
        _api_get_call(node, task_id, output_image_path, wait=False)
