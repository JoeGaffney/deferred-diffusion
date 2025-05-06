import nuke

from config import client
from generated.api_client.api.images import images_create
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET, Unset
from utils import base64_to_image, get_node_value, image_to_base64


def create_dd_image_node():
    # Create the node from the gizmo (no need to re-define defaults)
    node = nuke.createNode("dd_image")  # 'dd_image' is the name of your gizmo

    # Optionally: You can set other properties or interact with the node here
    # e.g., If you want to call a function defined inside the gizmo, you can do it here

    return node


def process_image(node):
    # This function is defined to process the node
    nuke.message("Processing image...")
    nuke.tprint(node)

    # Example of manipulating the node's properties
    print(f"Processing image for node: {node.name()}")
    start_frame = nuke.frame()
    end_frame = nuke.frame()

    output_image_path = get_node_value(node, "file", mode="evaluate")
    if not output_image_path:
        raise ValueError("Output image path is required.")

    output_read = nuke.toNode(f"{node.name()}.output_read")
    if not output_read:
        raise ValueError("Output read node not found.")

    image_node = node.input(0)
    mask_node = node.input(1)

    print(f"Image node: {image_node}")
    tmp_image = ""
    if image_node:
        nuke.render("image_write", start_frame, end_frame)
        tmp_image = get_node_value(node, "tmp_image", default="", mode="evaluate")

    print(f"Mask node: {mask_node}")
    tmp_mask = ""
    if mask_node:
        nuke.render("mask_write", start_frame, end_frame)
        tmp_mask = get_node_value(node, "tmp_mask", default="", mode="evaluate")
        if tmp_mask:
            tmp_image = tmp_mask

    body = ImageRequest(
        model=ImageRequestModel(get_node_value(node, "model", "sd1.5", mode="value")),
        # controlnets=get_control_nets(node),
        # guidance_scale=params.get("guidance_scale", Unset),
        image=image_to_base64(tmp_image),
        # inpainting_full_image=params.get("inpainting_full_image", False),
        # ip_adapters=get_ip_adapters(node),
        mask=image_to_base64(tmp_mask),
        # max_height=params.get("max_height", Unset),
        # max_width=params.get("max_width", Unset),
        # negative_prompt=params.get("negative_prompt", Unset),
        # num_inference_steps=params.get("num_inference_steps", Unset),
        # optimize_low_vram=params.get("optimize_low_vram", Unset),
        prompt=get_node_value(node, "prompt", UNSET, mode="get"),
        # seed=params.get("seed", Unset),
        # strength=params.get("strength", Unset),
        max_height=1024,
        max_width=1024,
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
    if output_read:
        nuke.message(f"Refreshing output read node: {output_read.name()}")
        output_read["reload"].execute()

    # Example of manipulating the node's properties
