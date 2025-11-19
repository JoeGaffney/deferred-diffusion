import time

import hou
from httpx import RemoteProtocolError

from config import client
from dd_text import prompt_optimizer
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
def _api_get_call(node, id, output_path: str, expanded_path: str, iterations=100, sleep_time=10):
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
        last_image = input_to_base64(node, "last_image")

        # NOTE park video for now
        # video = params.get("video", UNSET)
        # video_base64 = UNSET
        # if video and video != UNSET and video != "":
        #     # Check if the video file exists
        #     if not os.path.exists(video):
        #         raise ValueError(f"Video file does not exist: {video}")
        #     video_base64 = input_to_base64(video)

        body = VideoRequest(
            model=VideoRequestModel(params.get("model", UNSET)),
            image=image,
            last_image=last_image,
            prompt=params.get("prompt", ""),
            seed=params.get("seed", 0),
            num_frames=params.get("num_frames", UNSET),
            width=params.get("width", UNSET),
            height=params.get("height", UNSET),
        )

        _api_call(node, body, output_video_path)


def get_video(node):
    with houdini_error_handling(node):
        task_id = node.parm("task_id").eval()
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the video.")

        output_path = get_output_path(node, movie=True)
        _api_get_call(node, task_id, output_path, hou.expandString(output_path), iterations=1, sleep_time=0)


def video_prompt_optimizer(node):
    params = get_node_parameters(node)
    model = params.get("model", "sd-xl")
    prompt = params.get("prompt", "")
    image = input_to_base64(node, "src")
    last_image = input_to_base64(node, "last_image")

    system_prompt = (
        "You are an expert AI video prompt optimizer. Given a basic prompt and optional reference images, "
        "generate a structured prompt suitable for a single shot AI video generation. "
        f"The model the prompt is intended for is {model}. "
        ""
        "Your response must follow this template strictly, with each category separated by a new line, "
        "and keep each category as concise as possible: \n"
        "Action/Events: movement, events, progression\n"
        "Camera/Movement: perspective, angles, motion, transitions\n"
        "Environment/Setting: locations, time of day, atmosphere\n"
        "Style/Lighting/Rendering: visual style, lighting, color palette\n"
        "Characters/Objects: appearance and interactions\n"
        "Use reference images only as inspiration, not literal replication. "
        "Include cinematic and temporal keywords (e.g., slow motion, tracking shot, fade) where relevant. "
        ""
        "If multiple images are provided, treat the first as the start and the last as the end of the shot; "
        "use intermediates to inform transitions. If only one image is provided, use it for style, setting, and characters. "
        "If no images are provided, rely on the text prompt alone. "
        "Output only the optimized prompt, with no extra commentary."
    )

    images = []
    if image:
        images.append(image)

    if last_image:
        images.append(last_image)

    prompt_optimizer(node, prompt, system_prompt, images)
