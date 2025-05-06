import base64
import io
import os
from typing import Literal, Optional

from config import MAX_ADDITIONAL_IMAGES
from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.control_net_schema_model import ControlNetSchemaModel
from generated.api_client.models.ip_adapter_model import IpAdapterModel
from generated.api_client.models.ip_adapter_model_model import IpAdapterModelModel

# def save_tmp_image(node, node_name):
#     tmp_image_node = node.node(node_name)
#     if tmp_image_node is None:
#         return

#     print(f"Saving temporary image for node: {tmp_image_node.name()}")
#     try:
#         tmp_image_node.parm("execute").pressButton()  # Trigger execution
#     except Exception as e:
#         hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


# def save_all_tmp_images(node):
#     # Get all ROP image nodes from children
#     rop_nodes = [child for child in node.children() if child.type().name() == "rop_image"]
#     for rop_node in rop_nodes:
#         save_tmp_image(node, rop_node.name())


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


def get_node_value(
    node, knob_name: str, default=None, return_type=str, mode: Literal["get", "value", "evaluate"] = "get"
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
