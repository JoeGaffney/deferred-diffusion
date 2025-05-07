import base64
import io
import os
import tempfile
from typing import Literal, Optional, Union

import nuke

from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.control_net_schema_model import ControlNetSchemaModel
from generated.api_client.models.ip_adapter_model import IpAdapterModel
from generated.api_client.models.ip_adapter_model_model import IpAdapterModelModel
from generated.api_client.types import UNSET


def get_node_value(
    node,
    knob_name: str,
    default=None,
    return_type: type = str,
    mode: Literal["get", "value", "evaluate"] = "get",
):
    """Get the value of a knob from a node."""
    knob = node.knob(knob_name)
    if knob is None:
        return default

    if hasattr(knob, "value") is False:
        return default

    value = default
    if mode == "get":
        value = knob.getValue()
    elif mode == "value":
        value = knob.value()
    elif mode == "evaluate":
        value = knob.evaluate()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Knob value: {knob} - {value}")

    if isinstance(value, return_type):
        return value

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


def base64_to_image(base64_str: str, output_path: str, create_dir: bool = True):
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


def node_to_base64(input_node, current_frame):
    """Convert a Nuke node's output directly to base64 without saving to disk"""
    if not input_node:
        return None

    # Create a temporary Write node
    temp_write = nuke.nodes.Write(name="temp_write_to_base64")
    temp_write.setInput(0, input_node)
    temp_write["file_type"].setValue("png")

    temp_path = tempfile.NamedTemporaryFile(prefix="nuke_dd_tmp_", suffix=".png", delete=False).name
    temp_path = temp_path.replace("\\", "/")  # Convert backslashes to forward slashes
    temp_write["file"].setValue(temp_path)

    nuke.render(temp_write.name(), current_frame, current_frame)
    # Render the current frame
    # nuke.execute(temp_write.name(), current_frame, current_frame)

    print(f"Temporary image saved to: {temp_path}")
    result = image_to_base64(temp_path)

    # Clean up
    nuke.delete(temp_write)
    # os.remove(temp_path)

    return result


def get_control_nets(node) -> list[ControlNetSchema]:
    if node is None:
        return []

    current_frame = nuke.frame()

    # only the control_net nodes are valid inputs
    valid_inputs = []
    if node.Class() == "dd_controlnet":
        valid_inputs.append(node)
    else:
        print(f"Invalid node type. Expected 'dd_controlnet'. {node.Class()}")

    result = []
    for current in valid_inputs:

        image_node = current.input(0)
        image = node_to_base64(image_node, current_frame)
        if image is None:
            continue

        tmp = ControlNetSchema(
            model=ControlNetSchemaModel(get_node_value(current, "model", "depth", mode="value")),
            image=image,
            conditioning_scale=get_node_value(node, "conditioning_scale", UNSET, return_type=float, mode="value"),
        )
        result.append(tmp)

    return result


# Get the parameter template group
def get_ip_adapters(node) -> list[IpAdapterModel]:
    if node is None:
        return []

    current_frame = nuke.frame()

    # only the dd_adapter nodes are valid inputs
    valid_inputs = []
    if node.Class() == "dd_adapter":
        valid_inputs.append(node)
    else:
        print(f"Invalid node type. Expected 'dd_adapter'. {node.Class()}")

    result = []
    for current in valid_inputs:

        image_node = current.input(0)
        image = node_to_base64(image_node, current_frame)
        if image is None:
            print("Image is None")
            continue

        mask_node = current.input(1)
        mask = node_to_base64(mask_node, current_frame)

        tmp = IpAdapterModel(
            model=IpAdapterModelModel(get_node_value(current, "model", "style", mode="value")),
            image=image,
            mask=mask,
            scale=get_node_value(node, "scale", UNSET, return_type=float, mode="value"),
            scale_layers=get_node_value(current, "scale_layers", "all", mode="value"),
        )

        result.append(tmp)

    return result
