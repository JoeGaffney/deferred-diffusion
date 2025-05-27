import hou

from config import client
from generated.api_client.api.images import images_create, images_get
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_request_target_precision import (
    ImageRequestTargetPrecision,
)
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.types import UNSET
from utils import (
    base64_to_image,
    get_control_nets,
    get_ip_adapters,
    get_node_parameters,
    get_output_path,
    input_to_base64,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def api_get_call(node, id, output_path: str, wait=False):
    set_node_info(node, "PENDING", "")

    try:
        parsed = images_get.sync(id, client=client, wait=wait)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        hou.ui.displayMessage(f"Failed. {str(e)}")
        return

    def update_ui():
        if not isinstance(parsed, ImageResponse):
            message = "Unexpected response type from API call."
            set_node_info(node, "ERROR", message)
            hou.ui.displayMessage(message)
            return

        if not parsed.status == "SUCCESS" or not parsed.result:
            message = f"Task {parsed.status} with error: {parsed.error_message}"
            set_node_info(node, parsed.status, message)  # Use status from exception
            hou.ui.displayMessage(message)
            return

        # Save the image to the specified path before reloading the outputs
        resolved_output_path = hou.expandString(output_path)
        base64_to_image(parsed.result.base64_data, resolved_output_path, save_copy=True)

        node.parm("output_image_path").set(output_path)
        reload_outputs(node, "output_read")
        set_node_info(node, "COMPLETE", output_path)

    hou.ui.postEventCallback(update_ui)


def api_call(node, body: ImageRequest, output_image_path: str):
    try:
        parsed = images_create.sync(client=client, body=body)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        raise

    if not isinstance(parsed, ImageCreateResponse):
        set_node_info(node, "ERROR", "Unexpected response type from API call.")
        raise ValueError("Unexpected response type from API call.")

    try:
        node.parm("task_id").set(str(parsed.id))
    except:
        pass

    api_get_call(node, str(parsed.id), output_image_path, wait=True)


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
        target_precision=ImageRequestTargetPrecision(int(params.get("target_precision", 8))),
        prompt=params.get("prompt", UNSET),
        seed=params.get("seed", UNSET),
        strength=params.get("strength", UNSET),
    )

    api_call(node, body, output_image_path)


def main_get(node):
    task_id = node.parm("task_id").eval()
    if not task_id or task_id == "":
        hou.ui.displayMessage("Task ID is required to get the image.")
        return

    output_image_path = get_output_path(node, movie=False)
    api_get_call(node, task_id, output_image_path, wait=False)


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
