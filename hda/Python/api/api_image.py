import hou

from config import client
from generated.api_client.api.images import images_create
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from utils import (
    extract_and_format_parameters,
    get_control_nets,
    get_ip_adapters,
    reload_outputs,
    save_all_tmp_images,
)


def main(node):
    # Get all ROP image nodes from children
    save_all_tmp_images(node)

    params = extract_and_format_parameters(node)
    valid_params = {k: v for k, v in params.items() if k in ImageRequest.__annotations__}
    model = ImageRequestModel(valid_params.get("model", "sd1.5"))
    valid_params["model"] = model

    body = ImageRequest(
        **valid_params,
        ip_adapters=get_ip_adapters(node),
        controlnets=get_control_nets(node),
    )

    # make the API call
    response = images_create.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, ImageResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    reload_outputs(node, "output_read")


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    print(f"Frame Range: {start_frame} - {end_frame}")

    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
