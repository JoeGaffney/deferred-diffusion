import base64
import io
import os
import threading
from typing import Optional

import hou

from config import MAX_ADDITIONAL_IMAGES
from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.control_net_schema_model import ControlNetSchemaModel
from generated.api_client.models.ip_adapter_model import IpAdapterModel
from generated.api_client.models.ip_adapter_model_model import IpAdapterModelModel
from generated.api_client.types import Response


# Decorators
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    return wrapper


def handle_api_response(response: Response, expected_type, error_prefix="API Call Failed"):
    """Validates API response and returns parsed data or None if validation fails."""
    if response.status_code != 200:
        raise ApiResponseError(f"{error_prefix}: {response}", status_code=response.status_code, response=response)

    if not isinstance(response.parsed, expected_type):
        raise ApiResponseError(
            f"Invalid response type {type(expected_type)}: {type(response.parsed)}", response=response
        )

    return response.parsed


class ApiResponseError(Exception):
    """Exception raised for API response errors."""

    def __init__(self, message, status="FAILED", status_code=None, response=None):
        self.message = message
        self.status = status
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


def set_node_info(node, status, message, id=None):
    if id:
        node.setUserData("nodeinfo_api_id", str(id))

    node.setUserData("nodeinfo_api_status", str(status))
    node.setUserData("nodeinfo_api_message", str(message))

    if status == "COMPLETE":
        node.setColor(hou.Color((0.0, 0.8, 0.0)))
    elif status == "PENDING":
        node.setColor(hou.Color((0.5, 0.5, 0.0)))
    elif status == "FAILED" or status == "ERROR":
        node.setColor(hou.Color((0.8, 0.0, 0.0)))
    else:
        node.setColor(hou.Color((0.5, 0.5, 0.5)))


def save_tmp_image(node, node_name):
    tmp_image_node = node.node(node_name)
    if tmp_image_node is None:
        return

    print(f"Saving temporary image for node: {tmp_image_node.name()}")
    try:
        tmp_image_node.parm("execute").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def save_all_tmp_images(node):
    # Get all ROP image nodes from children
    rop_nodes = [child for child in node.children() if child.type().name() == "rop_image"]
    for rop_node in rop_nodes:
        save_tmp_image(node, rop_node.name())


def get_input_cop_data_in_base64(node, input_name, fmt="PNG"):
    """
    Converts the image from a specified input of the node to a Base64-encoded string.
    """

    # Validate the input name
    valid_inputs = [i.outputLabel() for i in node.inputConnections()]
    if input_name not in valid_inputs:
        hou.ui.displayMessage(f"Input '{input_name}' does not exist on node {node.path()}.")
        return None

    # Find the connected COP node
    cop_node = None
    for i in node.inputConnections():
        if i.outputLabel() == input_name:
            cop_node = i.output()
            break

    if cop_node is None:
        hou.ui.displayMessage(f"No COP node connected to input '{input_name}' on node {node.path()}.")
        return None

    # Cook the COP node and get the image data
    try:
        cop_node.cook(force=True)
        img = cop_node.image()
        if img is None:
            raise ValueError("Image data is empty or failed to load.")
    except Exception as e:
        hou.ui.displayMessage(f"Failed to cook COP node or retrieve image: {str(e)}")
        return None

    # Convert the image to Base64
    try:
        buf = io.BytesIO()
        img.save(buf, fmt)
        buf.seek(0)  # Move to the beginning of the buffer to prepare for reading
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as e:
        hou.ui.displayMessage(f"Failed to convert image to Base64: {str(e)}")
        return None


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


def reload_outputs(node, node_name):
    tmp_image_node = node.node(node_name)
    if tmp_image_node is None:
        return

    try:
        tmp_image_node.parm("reload").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def get_node_parameters(node):
    """Extract all top-level parameters from the HDA."""
    if node is None:
        return {}

    params = {}
    for parm_tuple in node.parmTuples():
        values = [parm.eval() for parm in parm_tuple]
        params[parm_tuple.name()] = values[0] if len(values) == 1 else values
    return params


def extract_and_format_parameters(node):
    params = get_node_parameters(node)

    # Remove 'images' if not in valid_inputs
    valid_inputs = []
    for i in node.inputConnections():
        valid_inputs.append(i.outputLabel())

    key_map = {
        "mask": "input_mask_path",
        "src": "input_image_path",
    }
    for i in range(MAX_ADDITIONAL_IMAGES):
        key_map[f"image_{i}"] = f"image_{i}_path"

    for key, param_key in key_map.items():
        if key not in valid_inputs and param_key in params:
            params.pop(param_key)

    return params


def get_control_nets(node) -> list[ControlNetSchema]:
    # only the control_net nodes are valid inputs
    valid_inputs = []
    for i in node.inputs():
        if i:
            if i.type().name() == "deferred_diffusion::control_net":
                valid_inputs.append(i)

    result = []
    for current in valid_inputs:

        params = get_node_parameters(current)
        save_all_tmp_images(current)
        image = image_to_base64(params.get("image_path", ""))
        if image is None:
            continue

        tmp = ControlNetSchema(
            model=ControlNetSchemaModel(params.get("model", "")),
            image=image,
            conditioning_scale=params.get("conditioning_scale", 0.5),
        )
        result.append(tmp)

    return result


# Get the parameter template group
def get_ip_adapters(node) -> list[IpAdapterModel]:
    # only the adapter nodes are valid inputs
    valid_inputs = []
    for i in node.inputs():
        if i:
            if i.type().name() == "deferred_diffusion::ip_adapter":
                valid_inputs.append(i)

    result = []
    for current in valid_inputs:

        params = get_node_parameters(current)
        save_all_tmp_images(current)
        image = image_to_base64(params.get("image_path", ""))
        if image is None:
            continue

        tmp = IpAdapterModel(
            model=IpAdapterModelModel(params.get("model", "")),
            image=image,
            mask=image_to_base64(params.get("mask_path", "")),
            scale=params.get("scale", 0.5),
            scale_layers=params.get("scale_layers", "all"),
        )
        result.append(tmp)

    return result


def add_spare_params(node, prefix, params):
    parm_group = node.parmTemplateGroup()
    for param_name, param_value in params.items():
        # Skip if the parameter already exists
        unique_name = f"{prefix}_{param_name}"

        try:
            # Create string parameter with proper name and value
            parm_template = hou.StringParmTemplate(
                name=unique_name,
                label=unique_name,
                num_components=1,
                default_value=[str(param_value)],  # Convert value to string
            )
            parm_group.addParmTemplate(parm_template)
        except Exception as e:
            print(f"Error adding parameter {param_name}: {e}")

    # Apply the parameter template
    node.setParmTemplateGroup(parm_group)
