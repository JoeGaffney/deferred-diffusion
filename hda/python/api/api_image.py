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
    get_control_nets,
    get_ip_adapters,
    get_node_parameters,
    get_output_path,
    houdini_error_handling,
    input_to_base64,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def _api_get_call(node, id, output_path: str, wait=False):
    set_node_info(node, "PENDING", "")

    try:
        parsed = images_get.sync(id, client=client, wait=wait)
    except Exception as e:

        def handle_error(error=e):
            with houdini_error_handling(node):
                raise RuntimeError(f"API call failed: {str(error)}") from error

        hou.ui.postEventCallback(handle_error)
        return

    def update_ui():
        with houdini_error_handling(node):
            if not isinstance(parsed, ImageResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the image to the specified path before reloading the outputs
            resolved_output_path = hou.expandString(output_path)
            base64_to_image(parsed.result.base64_data, resolved_output_path, save_copy=True)

            node.parm("output_image_path").set(output_path)
            reload_outputs(node, "output_read")
            set_node_info(node, "COMPLETE", output_path)

    hou.ui.postEventCallback(update_ui)


def _api_call(node, body: ImageRequest, output_image_path: str):
    try:
        parsed = images_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, ImageCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    node.parm("task_id").set(str(parsed.id))
    _api_get_call(node, str(parsed.id), output_image_path, wait=True)


def main(node):
    set_node_info(node, "", "")
    with houdini_error_handling(node):
        params = get_node_parameters(node)
        output_image_path = get_output_path(node, movie=False)
        image = input_to_base64(node, "src")
        mask = input_to_base64(node, "mask")

        body = ImageRequest(
            model=ImageRequestModel(params.get("model", "sdxl")),
            # controlnets=get_control_nets(node),
            image=image,
            # ip_adapters=get_ip_adapters(node),
            mask=mask,
            height=params.get("height", UNSET),
            width=params.get("width", UNSET),
            prompt=params.get("prompt", UNSET),
            seed=params.get("seed", UNSET),
            strength=params.get("strength", UNSET),
        )

        _api_call(node, body, output_image_path)


def main_get(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the image.")

        output_image_path = get_output_path(node, movie=False)
        _api_get_call(node, task_id, output_image_path, wait=False)


def main_frame_range(node):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        main(node)
