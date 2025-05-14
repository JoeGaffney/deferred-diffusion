import datetime
from datetime import datetime

import hou

from config import client
from generated.api_client.api.images import images_create, images_get
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET
from utils import (
    ApiResponseError,
    base64_to_image,
    get_control_nets,
    get_ip_adapters,
    get_node_parameters,
    get_output_path,
    handle_api_response,
    input_to_base64,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def api_get_call(id, output_path: str, node):
    try:
        response = images_get.sync_detailed(id, client=client)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        hou.ui.displayMessage(str(e))
        return

    def update_ui():
        try:
            parsed = handle_api_response(response, ImageResponse, "API Get Call Failed")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ApiResponseError(
                    f"Task {parsed.status} with error: {parsed.error_message}", status=parsed.status
                )
        except ApiResponseError as e:
            set_node_info(node, e.status, str(e))  # Use status from exception
            hou.ui.displayMessage(str(e))
            return

        # Save the image to the specified path before reloading the outputs
        resolved_output_path = hou.expandString(output_path)
        base64_to_image(parsed.result.base64_data, resolved_output_path, save_copy=True)

        node.parm("output_image_path").set(output_path)
        reload_outputs(node, "output_read")
        set_node_info(node, "COMPLETE", output_path)

    hou.ui.postEventCallback(update_ui)


def main(node):
    set_node_info(node, "", "")

    params = get_node_parameters(node)
    output_image_path = get_output_path(node, movie=False)
    image = input_to_base64(node, "src")
    mask = input_to_base64(node, "mask")

    body = ImageRequest(
        model=ImageRequestModel(params.get("model", "sd1.5")),
        controlnets=get_control_nets(node),
        guidance_scale=params.get("guidance_scale", UNSET),
        image=image,
        inpainting_full_image=params.get("inpainting_full_image", False),
        ip_adapters=get_ip_adapters(node),
        mask=mask,
        max_height=params.get("max_height", UNSET),
        max_width=params.get("max_width", UNSET),
        negative_prompt=params.get("negative_prompt", UNSET),
        num_inference_steps=params.get("num_inference_steps", UNSET),
        optimize_low_vram=params.get("optimize_low_vram", UNSET),
        prompt=params.get("prompt", UNSET),
        seed=params.get("seed", UNSET),
        strength=params.get("strength", UNSET),
    )

    # make the initial API call to create the image task
    id = None
    try:
        response = images_create.sync_detailed(client=client, body=body)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        raise

    try:
        parsed = handle_api_response(response, ImageCreateResponse)

        id = parsed.id
        if not id:
            raise ApiResponseError("No ID found in the response.")
    except ApiResponseError as e:
        set_node_info(node, e.status, str(e))  # Use status from exception
        raise

    set_node_info(node, "PENDING", "", str(id))
    api_get_call(id, output_image_path, node)


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
