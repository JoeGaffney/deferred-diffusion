import time

import hou
from config import client
from generated.api_client.api.image import create_image
from generated.api_client.models import ImageRequest, ImageResponse
from utils import (
    add_call_metadata,
    extract_and_format_parameters,
    get_control_nets,
    reload_outputs,
    save_tmp_image,
)


def main(node):
    # Get all ROP image nodes from children
    rop_nodes = [child for child in node.children() if child.type().name() == "rop_image"]
    for rop_node in rop_nodes:
        save_tmp_image(node, rop_node.name())

    params = extract_and_format_parameters(node)
    params["controlnets"] = get_control_nets(params)
    valid_params = {k: v for k, v in params.items() if k in ImageRequest.__annotations__}
    body = ImageRequest(**valid_params)

    # make the API call
    start_time = time.time()
    response = create_image.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, ImageResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    reload_outputs(node, "output_read")

    # apply back to the node
    add_call_metadata(node, body.to_dict(), response.parsed.to_dict(), start_time)


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    print(f"Frame Range: {start_frame} - {end_frame}")

    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
