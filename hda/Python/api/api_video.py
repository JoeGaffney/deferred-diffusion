import hou

from config import client
from generated.api_client.api.videos import videos_create
from generated.api_client.models import VideoRequest, VideoResponse
from generated.api_client.models.video_request_model import VideoRequestModel
from generated.api_client.types import Unset
from utils import (
    base64_to_image,
    extract_and_format_parameters,
    image_to_base64,
    reload_outputs,
    save_tmp_image,
)


def main(node):
    # gather our parameters and save any temporary images
    save_tmp_image(node, "tmp_input_image")

    params = extract_and_format_parameters(node)
    output_video_path = params.get("output_video_path", Unset)
    if not output_video_path:
        raise ValueError("Output image path is required.")

    image = image_to_base64(params.get("input_image_path", ""))
    if not image:
        raise ValueError("Input image is required.")

    body = VideoRequest(
        model=VideoRequestModel(params.get("model", "LTX-Video")),
        image=image,
        prompt=params.get("prompt", ""),
        seed=params.get("seed", 0),
        negative_prompt=params.get("negative_prompt", Unset),
        num_frames=params.get("num_frames", Unset),
        num_inference_steps=params.get("num_inference_steps", Unset),
        guidance_scale=params.get("guidance_scale", Unset),
    )

    # make the API call
    response = videos_create.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, VideoResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    # Save the image to the specified path before reloading the outputs
    base64_to_image(response.parsed.base64_data, output_video_path)
    reload_outputs(node, "output_read_video")
