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

from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.control_net_schema_model import ControlNetSchemaModel
from generated.api_client.models.ip_adapter_model import IpAdapterModel
from generated.api_client.models.ip_adapter_model_model import IpAdapterModelModel
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


# Decorators
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    return wrapper


def set_node_info(node, status, message):
    # Update the node label to show current status
    status_text = f"[{status}]"
    node["label"].setValue(status_text)
    if message:
        node["label"].setValue(f"{status_text}\n{message}")

    if status == "COMPLETE":
        node["tile_color"].setValue(0x00CC00FF)  # Green
    elif status == "PENDING":
        node["tile_color"].setValue(0xCCCC00FF)  # Yellow
    elif status in ["RUNNING", "IN_PROGRESS", "STARTED"]:
        node["tile_color"].setValue(0x0000CCFF)  # Blue
    elif status == ["FAILED", "ERROR", "FAILURE"]:
        node["tile_color"].setValue(0xCC0000FF)  # Red
    else:
        node["tile_color"].setValue(0x888888FF)  # Grey


@contextmanager
def nuke_error_handling(node):
    try:
        yield
    except ValueError as e:
        set_node_info(node, "ERROR", str(e))
        nuke.message(str(e))
    except Exception as e:
        set_node_info(node, "ERROR", str(e))
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


def get_all_parent_nodes(node, stop_at_types: List, visited=None) -> list:
    """Recursively get all upstream/parent nodes of the given node"""
    if visited is None:
        visited = set()

    if node is None or node in visited:
        return []

    visited.add(node)
    parent_nodes = []

    # Check all inputs
    for i in range(node.inputs()):
        input_node = node.input(i)
        if input_node is not None:
            # Stop recursion if this node type is in stop_at_types
            if input_node.Class() in stop_at_types:
                continue

            parent_nodes.append(input_node)

            # Recursively get the parents of this input
            parent_nodes.extend(get_all_parent_nodes(input_node, stop_at_types, visited))

    return parent_nodes


# pylint: disable=dangerous-default-value
def find_nodes_of_type(node, target_class_type: str, stop_at_types=[NODE_IMAGE]) -> list:
    """Find all nodes of a specific type in the current node and its parents."""
    if node is None:
        return []

    matching_nodes = []
    all_nodes = [node] + get_all_parent_nodes(node, stop_at_types, visited=None)

    # Filter only the nodes of the desired type
    for current in all_nodes:
        if current.Class() == target_class_type:
            matching_nodes.append(current)

    return matching_nodes


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
        # NOTE test excute Render the current frame
        nuke.execute(temp_write.name(), current_frame, current_frame)
        # nuke.render(temp_write.name(), current_frame, current_frame)
        nuke.tprint(f"Temporary image saved to: {temp_path}")

        result = image_to_base64(temp_path)
    except Exception as e:
        nuke.tprint(f"Error converting node {input_node.name()} to base64: {str(e)}")
        result = None

    # Clean up
    nuke.delete(temp_write)
    return result


def get_control_nets(node) -> list[ControlNetSchema]:
    if node is None:
        return []

    current_frame = nuke.frame()
    valid_inputs = find_nodes_of_type(node, "dd_controlnet")
    result = []
    for current in valid_inputs:

        image_node = current.input(0)
        image = node_to_base64(image_node, current_frame)
        if image is None:
            continue

        tmp = ControlNetSchema(
            model=ControlNetSchemaModel(get_node_value(current, "model", "depth", mode="value")),
            image=image,
            conditioning_scale=get_node_value(current, "conditioning_scale", UNSET, return_type=float, mode="value"),
        )
        result.append(tmp)

    return result


def get_ip_adapters(node) -> list[IpAdapterModel]:
    if node is None:
        return []

    current_frame = nuke.frame()
    valid_inputs = find_nodes_of_type(node, "dd_adapter")

    result = []
    for current in valid_inputs:

        image_node = current.input(0)
        image = node_to_base64(image_node, current_frame)
        if image is None:
            continue

        mask_node = current.input(1)
        mask = node_to_base64(mask_node, current_frame)

        tmp = IpAdapterModel(
            model=IpAdapterModelModel(get_node_value(current, "model", "style", mode="value")),
            image=image,
            mask=mask,
            scale=get_node_value(current, "scale", UNSET, return_type=float, mode="value"),
            scale_layers=get_node_value(current, "scale_layers", "all", mode="value"),
        )

        result.append(tmp)

    return result


def get_previous_text_messages(node) -> str:
    if node is None:
        return ""

    if node.Class() != "dd_text":
        return ""

    # NOTE some issues here with how nuke handles json encoded text so we read from text knob directly
    # k = node["chain_of_thought"]
    # chain_of_thought = k.value()
    # nuke.tprint(f"chain_of_thought: {chain_of_thought}")

    # k = node["chain_of_thought_alt"]
    # chain_of_thought_alt = k.value()
    # nuke.tprint(f"chain_of_thought_alt: {chain_of_thought_alt}")

    chain_of_thought = get_node_value(node, "chain_of_thought_alt", "[]", mode="get")
    nuke.tprint(f"chain_of_thought: {chain_of_thought}")
    return chain_of_thought


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
