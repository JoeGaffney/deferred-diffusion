import nuke

from config import client
from generated.api_client.api.images import images_create
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET, Unset
from utils import (
    base64_to_image,
    get_control_nets,
    get_ip_adapters,
    get_node_value,
    node_to_base64,
)


def create_dd_image_node():
    # Create the node from the gizmo (no need to re-define defaults)
    node = nuke.createNode("dd_image")  # 'dd_image' is the name of your gizmo

    # Optionally: You can set other properties or interact with the node here
    # e.g., If you want to call a function defined inside the gizmo, you can do it here

    return node


def process_image(node):
    # This function is defined to process the node
    # nuke.message("Processing image...")
    # nuke.tprint(node)

    output_image_path = get_node_value(node, "file", mode="evaluate")
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
        max_height=int(width_height[0]),
        max_width=int(width_height[1]),
    )

    # make the API call
    response = images_create.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        nuke.message(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, ImageResponse):
        nuke.message(f"Invalid response type: {type(response.parsed)} {response}")
        return

    base64_to_image(response.parsed.base64_data, output_image_path)
    output_read["reload"].execute()
