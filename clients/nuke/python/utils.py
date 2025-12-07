import base64
import os
import shutil
import tempfile
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import List, Literal, Optional

import nuke

from generated.api_client.models import References, ReferencesMode, TaskStatus
from generated.api_client.types import UNSET

NODE_CONTROLNET = "dd_controlnet"
NODE_ADAPTER = "dd_adapter"
NODE_IMAGE = "dd_image"

# Access mode constants
MODE_GET = "get"
MODE_VALUE = "value"
MODE_EVALUATE = "evaluate"

# Define the type for access mode
KnobAccessMode = Literal["get", "value", "evaluate"]


COMPLETED_STATUS = [
    TaskStatus.SUCCESS,
    TaskStatus.FAILURE,
    TaskStatus.REVOKED,
    TaskStatus.REJECTED,
    TaskStatus.IGNORED,
]


# Decorators
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    return wrapper


def set_node_info(node, status: Optional[TaskStatus], message: str = "", logs=None):
    # Update the node label to show current status
    if status is None:
        status_text = "[]"
    else:
        status_text = f"[{status}]"

    label_test = status_text
    if message:
        label_test = f"{status_text}\n{message}"

    if logs:
        # Join logs into a single string separated by newlines
        if len(logs) > 0:
            logs_text = "\n".join(logs)
            label_test += f"\nLogs:{logs_text}"

    node["label"].setValue(label_test)

    if status == TaskStatus.SUCCESS:
        node["tile_color"].setValue(0x00CC00FF)  # Green
    elif status == TaskStatus.PENDING:
        node["tile_color"].setValue(0xCCCC00FF)  # Yellow
    elif status in [TaskStatus.STARTED]:
        node["tile_color"].setValue(0x0000CCFF)  # Blue
    elif status in [TaskStatus.FAILURE]:
        node["tile_color"].setValue(0xCC0000FF)  # Red
    else:
        node["tile_color"].setValue(0x888888FF)  # Grey


def polling_message(count, iterations, sleep_time):
    current_time = count * sleep_time
    return f"🔄 {count}/{iterations} • {current_time}s"


@contextmanager
def nuke_error_handling(node):
    try:
        yield
    except ValueError as e:
        set_node_info(node, TaskStatus.FAILURE, str(e))
        nuke.message(str(e))
    except Exception as e:
        set_node_info(node, TaskStatus.FAILURE, str(e))
        traceback.print_exc()
        nuke.message(str(e))


def get_tmp_dir(user_tmp=False) -> str:
    if user_tmp:
        # Use the user's temporary directory
        subdir = tempfile.gettempdir()
        subdir = os.path.join(tempfile.gettempdir(), "deferred-diffusion")
    else:
        script_dir = os.path.dirname(nuke.root().name())
        subdir = os.path.join(script_dir, "deferred-diffusion", "tmp")

    os.makedirs(subdir, exist_ok=True)
    return subdir


def get_node_root_path(node):
    node_name = node.name()
    script_dir = os.path.dirname(nuke.root().name())
    path = f"{script_dir}/deferred-diffusion/{node_name}"
    os.makedirs(path, exist_ok=True)
    return path


def get_output_path(node, movie=False) -> str:
    root_path = get_node_root_path(node)
    time_stamp = str(node.__hash__())  # time_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    extension = "#####.png"
    if movie:
        extension = "mp4"

    output_image_path = f"{root_path}/{time_stamp}.{extension}"
    return output_image_path


def set_node_value(node, knob_name: str, value):
    """Set the value of a knob on a node."""
    knob = node.knob(knob_name)
    if knob is None:
        raise ValueError(f"Knob '{knob_name}' not found on node '{node.name()}'")

    if hasattr(knob, "setValue") is False:
        raise ValueError(f"Knob '{knob_name}' does not support setting value")

    try:
        knob.setValue(value)
    except Exception as e:
        raise ValueError(f"Failed to set knob '{knob_name}': {str(e)}") from e


def get_node_value(
    node,
    knob_name: str,
    default=None,
    return_type: type = str,
    mode: KnobAccessMode = MODE_GET,
):
    """Get the value of a knob from a node."""
    knob = node.knob(knob_name)
    if knob is None:
        raise ValueError(f"Knob '{knob_name}' not found on node '{node.name()}'")

    if hasattr(knob, "value") is False:
        raise ValueError(f"Knob '{knob_name}' does not support getting value")

    value = default
    if mode == MODE_GET:
        value = knob.getValue()
    elif mode == MODE_VALUE:
        value = knob.value()
    elif mode == MODE_EVALUATE:
        value = knob.evaluate()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if isinstance(value, return_type):
        return value

    try:
        return return_type(value)  # General casting approach
    except (ValueError, TypeError):
        return default


def image_to_base64(image_path: str, debug=False) -> Optional[str]:
    """Convert an image file to a base64 string (binary data encoded in base64)."""
    if not image_path:
        return None

    if not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

            # Convert the bytes to Base64 encoding (standard base64 encoding)
            base64_bytes = base64.b64encode(image_bytes)  # Get base64 as bytes
            base64_str = base64_bytes.decode("utf-8")  # Convert to a string

            # NOTE to debug: print the first 100 characters of the base64 string
            if debug:
                print(f"Base64 string: {base64_str[:100]}...")

            return base64_str
    except Exception as e:
        raise ValueError(f"Error encoding image {image_path}: {str(e)}") from e


def base64_to_file(base64_str: str, output_path: str, create_dir: bool = True):
    """Convert a base64 string to an image and save it to the specified path."""
    try:
        # Handle both string and bytes input
        if isinstance(base64_str, str):
            # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
            if "," in base64_str and ";base64," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            # Convert string to bytes if needed
            base64_bytes = base64_str.encode("utf-8")
        else:
            base64_bytes = base64_str

        # Decode the base64 to binary
        image_bytes = base64.b64decode(base64_bytes)

        # Create directory if it doesn't exist and create_dir is True
        dir_path = os.path.dirname(output_path)
        if create_dir and dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Write the bytes to the specified file path
        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)

    except Exception as e:
        raise ValueError(f"Error saving base64 to image {output_path}: {str(e)}") from e

    # save a timestamped version by copying the file to the sample path prepended with a /tmp/timestamp
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(output_path)
        current_dir = os.path.dirname(output_path)
        snapshot_dir = f"{current_dir}/snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        tmp_path = os.path.join(snapshot_dir, f"{timestamp}_{filename}")

        # Copy the file to the timestamped location
        shutil.copy2(output_path, tmp_path)
    except Exception as e:
        raise ValueError(f"Error creating snapshot of file {output_path}: {str(e)}") from e


def node_to_base64(input_node, current_frame):
    """Convert a Nuke node's output directly to base64 without saving to disk"""
    if not input_node:
        return None

    # maybe its better to keep with the node in the current directory
    tmp_dir = get_tmp_dir()

    # Create a temporary Write node
    temp_write = nuke.nodes.Write(name="temp_write_to_base64")
    temp_write.setInput(0, input_node)
    temp_write["file_type"].setValue("png")

    temp_path = tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".png", delete=False).name
    temp_path = temp_path.replace("\\", "/")  # Convert backslashes to forward slashes
    temp_write["file"].setValue(temp_path)
    nuke.tprint(f"Temporary image path: {temp_path} - {temp_write.name()}")

    try:
        nuke.execute(temp_write.name(), current_frame, current_frame)
        nuke.tprint(f"Temporary image saved to: {temp_path}")

        result = image_to_base64(temp_path)
    except Exception as e:
        nuke.tprint(f"Error converting node {input_node.name()} to base64: {str(e)}")
        result = None

    # Clean up
    nuke.delete(temp_write)
    return result


def node_to_base64_video(input_node, current_frame, num_frames=24):
    """Convert a Nuke node's output directly to base64 H.264 video"""
    if not input_node:
        return None

    tmp_dir = get_tmp_dir()

    # Create a temporary Write node for video
    temp_write = nuke.nodes.Write(name="temp_write_to_base64_video")
    temp_write.setInput(0, input_node)
    temp_write["file_type"].setValue("mov")
    temp_write["mov64_codec"].setValue("h264")

    temp_path = tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".mp4", delete=False).name
    temp_path = temp_path.replace("\\", "/")
    temp_write["file"].setValue(temp_path)
    nuke.tprint(f"Temporary video path: {temp_path} - {temp_write.name()}")

    try:
        # Calculate frame range
        start_frame = current_frame
        end_frame = current_frame + num_frames - 1

        nuke.execute(temp_write.name(), start_frame, end_frame)
        nuke.tprint(f"Temporary video saved to: {temp_path}")

        result = image_to_base64(temp_path)
    except Exception as e:
        nuke.tprint(f"Error converting node {input_node.name()} to base64 video: {str(e)}")
        result = None

    # Clean up
    nuke.delete(temp_write)
    return result


def get_references(node, start_index=2) -> list[References]:
    if node is None:
        return []

    current_frame = nuke.frame()
    result = []

    # starts at 2 because 0 is main image, 1 is mask
    index = start_index
    for current in ["reference_a", "reference_b", "reference_c"]:
        image_node = node.input(index)
        index += 1
        image = node_to_base64(image_node, current_frame)
        if image is None:
            continue

        tmp = References(
            mode=ReferencesMode(get_node_value(node, f"{current}_mode", "style", mode="value")),
            strength=get_node_value(node, f"{current}_strength", UNSET, return_type=float, mode="value"),
            image=image,
        )
        result.append(tmp)

    return result


def replace_hashes_with_frame(path_with_hashes, frame):
    num_hashes = path_with_hashes.count("#")
    if num_hashes == 0:
        # No hashes to replace, return original path
        return path_with_hashes
    frame_str = str(frame).zfill(num_hashes)
    return path_with_hashes.replace("#" * num_hashes, frame_str)


def update_read_range(read_node):
    read_node["reload"].execute()
    read_node["first"].setValue(int(read_node.firstFrame()))
    read_node["last"].setValue(int(read_node.lastFrame()))
