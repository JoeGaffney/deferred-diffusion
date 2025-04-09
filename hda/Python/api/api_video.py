import time

import hou
from config import client
from generated.api_client.api.video import create_video
from generated.api_client.models import VideoRequest, VideoResponse
from utils import (
    add_call_metadata,
    extract_and_format_parameters,
    reload_outputs,
    save_tmp_image,
)


def main(node):
    # gather our parameters and save any temporary images
    save_tmp_image(node, "tmp_input_image")

    params = extract_and_format_parameters(node)
    valid_params = {k: v for k, v in params.items() if k in VideoRequest.__annotations__}
    body = VideoRequest(**valid_params)

    # make the API call
    start_time = time.time()
    response = create_video.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, VideoResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    reload_outputs(node, "output_read_video")

    # apply back to the node
    add_call_metadata(node, body.to_dict(), response.parsed.to_dict(), start_time)
