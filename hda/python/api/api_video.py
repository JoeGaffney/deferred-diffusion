import hou

from config import client
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models import VideoCreateResponse, VideoRequest, VideoResponse
from generated.api_client.models.comfy_workflow import ComfyWorkflow
from generated.api_client.models.video_request_model import VideoRequestModel
from generated.api_client.types import UNSET
from utils import (
    ApiResponseError,
    base64_to_image,
    get_node_parameters,
    get_output_path,
    handle_api_response,
    input_to_base64,
    load_comfy_workflow,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def api_get_call(id, output_path: str, node):
    try:
        response = videos_get.sync_detailed(id, client=client)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        hou.ui.displayMessage(str(e))
        return

    def update_ui():
        try:
            parsed = handle_api_response(response, VideoResponse, "API Get Call Failed")

            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ApiResponseError(
                    f"Task {parsed.status} with error: {parsed.error_message}", status=parsed.status
                )
        except ApiResponseError as e:
            set_node_info(node, e.status, str(e))
            hou.ui.displayMessage(str(e))
            return

        # Save the video to the specified path before reloading the outputs
        resolved_output_path = hou.expandString(output_path)
        base64_to_image(parsed.result.base64_data, resolved_output_path, save_copy=True)

        node.parm("output_video_path").set(output_path)
        reload_outputs(node, "output_read_video")
        set_node_info(node, "COMPLETE", output_path)

    hou.ui.postEventCallback(update_ui)


def main(node):
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

    # make the initial API call to create the video task
    id = None
    try:
        response = videos_create.sync_detailed(client=client, body=body)
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
        raise

    try:
        parsed = handle_api_response(response, VideoCreateResponse)

        id = parsed.id
        if not id:
            raise ApiResponseError("No ID found in the response.")
    except ApiResponseError as e:
        set_node_info(node, e.status, str(e))
        raise

    set_node_info(node, "PENDING", "")
    api_get_call(id, output_video_path, node)
