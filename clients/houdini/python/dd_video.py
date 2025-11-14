import time

import hou
from httpx import RemoteProtocolError

from config import client
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models import (
    TaskStatus,
    VideoCreateResponse,
    VideoRequest,
    VideoRequestModel,
    VideoResponse,
)
from generated.api_client.types import UNSET
from utils import (
    COMPLETED_STATUS,
    base64_to_image,
    get_node_parameters,
    get_output_path,
    houdini_error_handling,
    input_to_base64,
    polling_message,
    reload_outputs,
    set_node_info,
    threaded,
)


@threaded
def _api_get_call(node, id, output_path: str, expanded_path: str, iterations=100, sleep_time=5):
    set_node_info(node, TaskStatus.PENDING, "")

    for count in range(1, iterations + 1):
        time.sleep(sleep_time)

        try:
            parsed = videos_get.sync(id, client=client)
            if not isinstance(parsed, VideoResponse):
                break

            if parsed.status in COMPLETED_STATUS:
                break

            def progress_update(parsed=parsed, count=count):
                set_node_info(node, parsed.status, polling_message(count, iterations, sleep_time))

            hou.ui.postEventCallback(progress_update)
        except RemoteProtocolError:
            continue  # Retry on protocol errors attempt again
        except Exception as e:

            def handle_error(error=e):
                with houdini_error_handling(node):
                    raise RuntimeError(f"API call failed: {str(error)}") from error

            hou.ui.postEventCallback(handle_error)
            return

    def update_ui():
        with houdini_error_handling(node):
            if not isinstance(parsed, VideoResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the video to the specified path before reloading the outputs
            base64_to_image(parsed.result.base64_data, expanded_path, save_copy=True)

            node.parm("output_video_path").set(output_path)
            reload_outputs(node, "output_read_video")
            set_node_info(node, TaskStatus.SUCCESS, output_path)

    hou.ui.postEventCallback(update_ui)


def _api_call(node, body: VideoRequest, output_path: str):
    try:
        parsed = videos_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, VideoCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    node.parm("task_id").set(str(parsed.id))
    _api_get_call(node, str(parsed.id), output_path, hou.expandString(output_path), iterations=20, sleep_time=5)


def process_video(node):
    with houdini_error_handling(node):
        set_node_info(node, None, "")
        params = get_node_parameters(node)
        output_video_path = get_output_path(node, movie=True)
        image = input_to_base64(node, "src")
        if not image:
            raise ValueError("Input image is required.")

        body = VideoRequest(
            model=VideoRequestModel(params.get("model", UNSET)),
            image=image,
            prompt=params.get("prompt", ""),
            seed=params.get("seed", 0),
            num_frames=params.get("num_frames", UNSET),
        )

        _api_call(node, body, output_video_path)


def get_video(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the video.")

        output_path = get_output_path(node, movie=True)
        _api_get_call(node, task_id, output_path, hou.expandString(output_path), iterations=1, sleep_time=0)
