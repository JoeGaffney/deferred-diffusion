import os
import time

import nuke
from httpcore import RemoteProtocolError

from config import client
from dd_text import prompt_optimizer
from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.models import (
    TaskStatus,
    VideoCreateResponse,
    VideoRequest,
    VideoRequestModel,
)
from generated.api_client.models.video_response import VideoResponse
from generated.api_client.types import UNSET
from utils import (
    COMPLETED_STATUS,
    base64_to_file,
    get_node_value,
    get_output_path,
    image_to_base64,
    node_to_base64,
    nuke_error_handling,
    polling_message,
    set_node_info,
    set_node_value,
    threaded,
    update_read_range,
)


def create_dd_video_node():
    node = nuke.createNode("dd_video")
    return node


@threaded
def _api_get_call(node, id, output_path: str, current_frame: int, iterations=100, sleep_time=10):
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

            nuke.executeInMainThread(progress_update)
        except RemoteProtocolError:
            continue  # Retry on protocol errors attempt again
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
            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            # Save the movie to the specified path
            base64_to_file(parsed.result.base64_data, output_path)

            output_read = nuke.toNode(f"{node.name()}.output_read")
            set_node_value(output_read, "file", output_path)
            update_read_range(output_read)

            # Set the time offset to the current frame
            set_node_value(node, "time_offset", current_frame)
            set_node_info(node, TaskStatus.SUCCESS, "")

    nuke.executeInMainThread(update_ui)


def _api_call(node, body: VideoRequest, output_video_path: str, current_frame: int):
    try:
        parsed = videos_create.sync(client=client, body=body)
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}") from e

    if not isinstance(parsed, VideoCreateResponse):
        raise ValueError("Unexpected response type from API call.")

    set_node_value(node, "task_id", str(parsed.id))
    _api_get_call(node, str(parsed.id), output_video_path, current_frame)


def process_video(node):
    set_node_info(node, None, "")
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        output_video_path = get_output_path(node, movie=True)
        if not output_video_path:
            raise ValueError("Output video path is required.")

        image_node = node.input(0)
        last_image_node = node.input(1)
        image = node_to_base64(image_node, current_frame)
        last_image = node_to_base64(last_image_node, current_frame)

        # video input we extract from a file path parameter at the moment
        video = get_node_value(node, "video", UNSET, mode="get")
        nuke.tprint(f"Processing video: {video}")
        video_base64 = UNSET
        if video and video != UNSET and video != "":
            # Check if the video file exists
            if not os.path.exists(video):
                raise ValueError(f"Video file does not exist: {video}")
            video_base64 = image_to_base64(video)

        width_height = get_node_value(node, "width_height", [1280, 720], return_type=list, mode="value")

        body = VideoRequest(
            model=VideoRequestModel(get_node_value(node, "model", UNSET, mode="value")),
            image=image,
            last_image=last_image,
            video=video_base64,
            prompt=get_node_value(node, "prompt", UNSET, mode="get"),
            num_frames=get_node_value(node, "num_frames", UNSET, return_type=int, mode="value"),
            seed=get_node_value(node, "seed", UNSET, return_type=int, mode="value"),
            width=int(width_height[0]),
            height=int(width_height[1]),
        )

        _api_call(node, body, output_video_path, current_frame)


def get_video(node):
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the video.")

        output_video_path = get_output_path(node, movie=True)
        _api_get_call(node, task_id, output_video_path, current_frame, iterations=1, sleep_time=0)


def video_prompt_optimizer(node):
    current_frame = nuke.frame()
    model = get_node_value(node, "model", "wan-2", mode="value")
    prompt = get_node_value(node, "prompt", "", mode="get")
    image_node = node.input(0)
    last_image_node = node.input(1)

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
    image = node_to_base64(image_node, current_frame)
    if image:
        images.append(image)

    last_image = node_to_base64(last_image_node, current_frame)
    if last_image:
        images.append(last_image)

    prompt_optimizer(node, prompt, system_prompt, images)
