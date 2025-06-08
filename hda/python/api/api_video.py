import hou

from config import client
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models import VideoCreateResponse, VideoRequest, VideoResponse
from generated.api_client.models.comfy_workflow import ComfyWorkflow
from generated.api_client.models.video_request_model import VideoRequestModel
from generated.api_client.types import UNSET
from utils import (
    base64_to_image,
    get_node_parameters,
    get_output_path,
    houdini_error_handling,
    input_to_base64,
    load_comfy_workflow,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def _api_get_call(node, id, output_path: str, wait=False):
    try:
        parsed = videos_get.sync(id, client=client, wait=wait)
    except Exception as e:

        def handle_error():
            with houdini_error_handling(node):
                raise RuntimeError(f"API call failed: {str(e)}") from e

        hou.ui.postEventCallback(handle_error)
        return

    def update_ui():
        with houdini_error_handling(node):
            if not isinstance(parsed, VideoResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the video to the specified path before reloading the outputs
            resolved_output_path = hou.expandString(output_path)
            base64_to_image(parsed.result.base64_data, resolved_output_path, save_copy=True)

            node.parm("output_video_path").set(output_path)
            reload_outputs(node, "output_read_video")
            set_node_info(node, "COMPLETE", output_path)

    hou.ui.postEventCallback(update_ui)


def _api_call(node, body: VideoRequest, output_image_path: str):
    try:
        parsed = videos_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, VideoCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    node.parm("task_id").set(str(parsed.id))
    _api_get_call(node, str(parsed.id), output_image_path, wait=True)


def main(node):
    with houdini_error_handling(node):
        set_node_info(node, "", "")
        params = get_node_parameters(node)
        output_video_path = get_output_path(node, movie=True)
        image = input_to_base64(node, "src")
        if not image:
            raise ValueError("Input image is required.")

        comfy_workflow = params.get("comfy_workflow", "")
        if comfy_workflow != "":
            workflow_dict = load_comfy_workflow(comfy_workflow)
            comfy_workflow = ComfyWorkflow.from_dict(workflow_dict)
        else:
            comfy_workflow = UNSET

        body = VideoRequest(
            model=VideoRequestModel(params.get("model", "LTX-Video")),
            comfy_workflow=comfy_workflow,
            image=image,
            prompt=params.get("prompt", ""),
            seed=params.get("seed", 0),
            negative_prompt=params.get("negative_prompt", UNSET),
            num_frames=params.get("num_frames", UNSET),
            num_inference_steps=params.get("num_inference_steps", UNSET),
            guidance_scale=params.get("guidance_scale", UNSET),
        )

        _api_call(node, body, output_video_path)


def main_get(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the image.")

        output_image_path = get_output_path(node, movie=False)
        _api_get_call(node, task_id, output_image_path, wait=False)
