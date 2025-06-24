import os

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


def replace_hashes_with_frame(path_with_hashes, frame):
    num_hashes = path_with_hashes.count("#")
    if num_hashes == 0:
        # No hashes to replace, return original path
        return path_with_hashes
    frame_str = str(frame).zfill(num_hashes)
    return path_with_hashes.replace("#" * num_hashes, frame_str)


@threaded
def api_get_call(id, output_image_path: str, output_read):
    nuke.tprint("Calling API get...")
    response = images_get.sync_detailed(id, client=client)

    def update_ui():
        if response.status_code != 200:
            nuke.message(f"API Call Failed: {response}")
            return

        if not isinstance(response.parsed, ImageResponse):
            nuke.message(f"Invalid response type: {type(response.parsed)} {response}")
            return

        if not response.parsed.result:
            nuke.message("No result found in the response.")
            return

        if not response.parsed.status == "SUCCESS":
            nuke.message(f"Task failed with error: {response.parsed.error_message}")
            return

        resolved_output_path = replace_hashes_with_frame(output_image_path, nuke.frame())
        nuke.tprint(f"resolved_output_path {resolved_output_path}")

        base64_to_image(response.parsed.result.base64_data, resolved_output_path)
        set_node_value(output_read, "file", output_image_path)
        output_read["reload"].execute()

    nuke.executeInMainThread(update_ui)
    nuke.tprint("API call completed successfully.")


def api_call(node, body: ImageRequest, output_image_path: str, output_read):
    nuke.tprint("Calling API...")
    response = images_create.sync_detailed(client=client, body=body)

    if response.status_code != 200:
        nuke.message(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, ImageCreateResponse):
        nuke.message(f"Invalid response type: {type(response.parsed)} {response}")
        return

    id = response.parsed.id
    if not id:
        nuke.message("No ID found in the response.")
        return

    set_node_value(node, "task_id", str(id))
    api_get_call(str(id), output_image_path, output_read)


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


def process_image(node):
    """This function is defined to process the node"""

    output_image_path = get_output_path(node, movie=False)
    if not output_image_path:
        raise ValueError("Output image path is required.")

    output_read = nuke.toNode(f"{node.name()}.output_read")
    if not output_read:
        raise ValueError("Output read node not found.")

    current_frame = nuke.frame()
    image_node = node.input(0)
    mask_node = node.input(1)
    controlnets_node = node.input(2)
    apdapter_node = node.input(3)

    image = node_to_base64(image_node, current_frame)
    mask = node_to_base64(mask_node, current_frame)
    width_height = get_node_value(node, "width_height", [1024, 1024], return_type=list, mode="value")

    body = ImageRequest(
        model=ImageRequestModel(get_node_value(node, "model", "sd1.5", mode="value")),
        controlnets=get_control_nets(controlnets_node),
        guidance_scale=get_node_value(node, "guidance_scale", UNSET, return_type=float, mode="value"),
        image=image,
        ip_adapters=get_ip_adapters(apdapter_node),
        mask=mask,
        negative_prompt=get_node_value(node, "negative_prompt", UNSET, mode="get"),
        num_inference_steps=get_node_value(node, "num_inference_steps", UNSET, return_type=int, mode="value"),
        prompt=get_node_value(node, "prompt", UNSET, mode="get"),
        seed=get_node_value(node, "seed", UNSET, return_type=int, mode="value"),
        strength=get_node_value(node, "strength", UNSET, return_type=float, mode="value"),
        width=int(width_height[0]),
        height=int(width_height[1]),
    )

    api_call(node, body, output_image_path, output_read)
