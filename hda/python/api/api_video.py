import hou

from config import client
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models import VideoCreateResponse, VideoRequest, VideoResponse
from generated.api_client.models.video_request_model import VideoRequestModel
from generated.api_client.types import UNSET
from utils import (
    base64_to_image,
    extract_and_format_parameters,
    handle_api_response,
    image_to_base64,
    reload_outputs,
    save_tmp_image,
    threaded,
)


@threaded
def api_get_call(id, output_video_path: str, node):
    response = videos_get.sync_detailed(id, client=client)

    def update_ui():
        parsed = handle_api_response(response, VideoResponse, "API Get Call Failed")
        if not parsed:
            return

        if not parsed.status == "SUCCESS":
            hou.ui.displayMessage(f"Task {parsed.status} with error: {parsed.error_message}")
            return

        if not parsed.result:
            hou.ui.displayMessage("No result found in the response.")
            return

        # Save the video to the specified path before reloading the outputs
        base64_to_image(parsed.result.base64_data, output_video_path)
        reload_outputs(node, "output_read_video")

    hou.ui.postEventCallback(update_ui)


def main(node):
    # gather our parameters and save any temporary images
    save_tmp_image(node, "tmp_input_image")

    params = extract_and_format_parameters(node)
    output_video_path = params.get("output_video_path", UNSET)
    if not output_video_path:
        raise ValueError("Output video path is required.")

    image = image_to_base64(params.get("input_image_path", ""))
    if not image:
        raise ValueError("Input image is required.")

    body = VideoRequest(
        model=VideoRequestModel(params.get("model", "LTX-Video")),
        image=image,
        prompt=params.get("prompt", ""),
        seed=params.get("seed", 0),
        negative_prompt=params.get("negative_prompt", UNSET),
        num_frames=params.get("num_frames", UNSET),
        num_inference_steps=params.get("num_inference_steps", UNSET),
        guidance_scale=params.get("guidance_scale", UNSET),
    )

    # make the initial API call to create the video task
    response = videos_create.sync_detailed(client=client, body=body)
    parsed = handle_api_response(response, VideoCreateResponse)
    if not parsed:
        return

    id = parsed.id
    if not id:
        hou.ui.displayMessage("No ID found in the response.")
        return

    api_get_call(id, output_video_path, node)
