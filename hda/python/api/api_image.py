import hou

from config import client
from generated.api_client.api.images import images_create, images_get
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET
from utils import (
    base64_to_image,
    extract_and_format_parameters,
    get_control_nets,
    get_ip_adapters,
    handle_api_response,
    image_to_base64,
    reload_outputs,
    save_all_tmp_images,
    threaded,
)


@threaded
def api_get_call(id, output_image_path: str, node):
    response = images_get.sync_detailed(id, client=client)

    def update_ui():
        parsed = handle_api_response(response, ImageResponse, "API Get Call Failed")
        if not parsed:
            return

        if not parsed.status == "SUCCESS":
            hou.ui.displayMessage(f"Task {parsed.status} with error: {parsed.error_message}")
            return

        if not parsed.result:
            hou.ui.displayMessage("No result found in the response.")
            return

        # Save the image to the specified path before reloading the outputs
        base64_to_image(parsed.result.base64_data, output_image_path)
        reload_outputs(node, "output_read")

    hou.ui.postEventCallback(update_ui)


def main(node):
    # Get all ROP image nodes from children
    save_all_tmp_images(node)

    params = extract_and_format_parameters(node)
    output_image_path = params.get("output_image_path", UNSET)
    if not output_image_path:
        raise ValueError("Output image path is required.")

    body = ImageRequest(
        model=ImageRequestModel(params.get("model", "sd1.5")),
        controlnets=get_control_nets(node),
        guidance_scale=params.get("guidance_scale", UNSET),
        image=image_to_base64(params.get("input_image_path", "")),
        inpainting_full_image=params.get("inpainting_full_image", False),
        ip_adapters=get_ip_adapters(node),
        mask=image_to_base64(params.get("input_mask_path", "")),
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
    response = images_create.sync_detailed(client=client, body=body)
    parsed = handle_api_response(response, ImageCreateResponse)
    if not parsed:
        return

    id = parsed.id
    if not id:
        hou.ui.displayMessage("No ID found in the response.")
        return

    api_get_call(id, output_image_path, node)


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
