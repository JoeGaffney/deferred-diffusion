import time

import nuke

from config import client
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models.video_create_response import VideoCreateResponse
from generated.api_client.models.video_request import VideoRequest
from generated.api_client.models.video_request_model import VideoRequestModel
from generated.api_client.models.video_response import VideoResponse
from generated.api_client.types import UNSET
from utils import (
    base64_to_file,
    get_node_value,
    get_output_path,
    node_to_base64,
    nuke_error_handling,
    set_node_info,
    set_node_value,
    threaded,
    update_read_range,
)


def create_dd_video_node():
    node = nuke.createNode("dd_video")
    return node


@threaded
def _api_get_call(node, id, output_path: str, current_frame: int, iterations=1, sleep_time=20):
    set_node_info(node, "PENDING", "")

    for count in range(1, iterations + 1):

        try:
            parsed = videos_get.sync(id, client=client)
            if not isinstance(parsed, VideoResponse):
                break
            if parsed.status in ["SUCCESS", "COMPLETED", "ERROR", "FAILED"]:
                break

            def progress_update():
                if isinstance(parsed, VideoResponse):
                    message = f"Polling attempt {count}/{iterations}"
                    print(message)
                    set_node_info(node, parsed.status, message)

            nuke.executeInMainThread(progress_update)
            time.sleep(sleep_time)
        except Exception as e:

            def handle_error(error=e):
                with nuke_error_handling(node):
                    raise RuntimeError(f"API call failed: {str(error)}") from error

            nuke.executeInMainThread(handle_error)
            return

    def update_ui():
        with nuke_error_handling(node):
            if not isinstance(parsed, VideoResponse):
                raise ValueError("Unexpected response type from API call.")
            if not parsed.status == "SUCCESS" or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the movie to the specified path
            base64_to_file(parsed.result.base64_data, output_path)

            output_read = nuke.toNode(f"{node.name()}.output_read")
            set_node_value(output_read, "file", output_path)
            update_read_range(output_read)

            # Set the time offset to the current frame
            set_node_value(node, "time_offset", current_frame)
            set_node_info(node, "COMPLETE", "")

    nuke.executeInMainThread(update_ui)


def _api_call(node, body: VideoRequest, output_video_path: str, current_frame: int):
    try:
        parsed = videos_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, VideoCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    set_node_value(node, "task_id", str(parsed.id))
    _api_get_call(node, str(parsed.id), output_video_path, current_frame, iterations=25)


def process_video(node):
    set_node_info(node, "", "")
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        output_video_path = get_output_path(node, movie=True)
        if not output_video_path:
            raise ValueError("Output video path is required.")

        image_node = node.input(0)
        image = node_to_base64(image_node, current_frame)
        if not image:
            raise ValueError("Image input is required.")

        image_last_frame_node = node.input(1)
        image_last_frame = node_to_base64(image_last_frame_node, current_frame)

        body = VideoRequest(
            model=VideoRequestModel(get_node_value(node, "model", "external-runway-gen-3", mode="value")),
            image=image,
            prompt=get_node_value(node, "prompt", UNSET, mode="get"),
            negative_prompt=get_node_value(node, "negative_prompt", UNSET, mode="get"),
            guidance_scale=get_node_value(node, "guidance_scale", UNSET, return_type=float, mode="value"),
            num_frames=get_node_value(node, "num_frames", UNSET, return_type=int, mode="value"),
            num_inference_steps=get_node_value(node, "num_inference_steps", UNSET, return_type=int, mode="value"),
            seed=get_node_value(node, "seed", UNSET, return_type=int, mode="value"),
            image_last_frame=image_last_frame,
        )
        _api_call(node, body, output_video_path, current_frame)


def get_video(node):
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the video.")

        output_video_path = get_output_path(node, movie=True)
        _api_get_call(node, task_id, output_video_path, current_frame, iterations=1, sleep_time=5)
