import base64
import io
import os
import shutil
import tempfile
import threading
from datetime import datetime
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


def get_tmp_dir() -> str:
    subdir = os.path.join(tempfile.gettempdir(), "deferred-diffusion")
    os.makedirs(subdir, exist_ok=True)
    return subdir


def get_output_path(node, movie=False) -> str:
    node_name = node.name()
    time_stamp = str(node.sessionId())
    # time_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    extension = "`padzero(5, $F)`.png"
    if movie:
        extension = "mp4"

    output_image_path = f"$HIP/deferred-diffusion/{node_name}/{time_stamp}.{extension}"
    return output_image_path


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
    elif status == "FAILED" or status == "ERROR" or status == "FAILURE":
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


def image_to_base64(image_path: str, debug=False) -> Optional[str]:
    """Convert an image file to a base64 string (binary data encoded in base64)."""
    if not image_path:
        return None

    if not os.path.exists(image_path):
        return None

    if os.stat(image_path).st_size < 1000:  # 1000 bytes is still tiny for a real image
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


def base64_to_image(base64_str: str, output_path: str, save_copy: bool = False):
    """Convert a base64 string to an image and save it to the specified path."""

    def save_copy_with_timestamp(path):
        if os.path.exists(path) and save_copy:
            directory, filename = os.path.split(path)

            # Create the timestamped path
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Keep only 3 digits of milliseconds
            timestamp_path = os.path.join(directory, timestamp, filename)
            dir_path = os.path.dirname(timestamp_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)

            shutil.copy(path, timestamp_path)

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
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Write the bytes to the specified file path
        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)

        save_copy_with_timestamp(output_path)
    except Exception as e:
        raise ValueError(f"Error saving base64 to image {output_path}: {str(e)}") from e


def input_to_base64(node, input_name):
    """Converts the image from a specified input of the node to a Base64-encoded string."""
    cop_node = None
    for i in node.inputConnections():
        if i.outputLabel() == input_name:
            cop_node = i.inputNode()
            break

    if cop_node is None:
        return None

    def find_top_copnet():
        node = hou.pwd()
        while node and node.type().name() != "copnet":
            node = node.parent()
        return node

    # Cook the COP node and get the image data
    try:
        cop_node.cook(force=True)
    except Exception as e:
        print(f"Failed to cook COP node: {cop_node.name()} {e}")
        return None

    # Create a temporary ROP to write the image
    tmp_name = f"tmp_{node.name()}_{input_name}_{cop_node.name()}"
    temp_path = tempfile.NamedTemporaryFile(dir=get_tmp_dir(), prefix=tmp_name, suffix=".png", delete=False).name
    rop = find_top_copnet().createNode("rop_image", tmp_name)

    rop.parm("coppath").set(cop_node.path())
    rop.parm("copoutput").set(temp_path)
    rop.parm("execute").pressButton()

    # Convert the saved file to base64
    result = image_to_base64(temp_path)

    # NOTE: keep the file for debugging
    # Clean up
    rop.destroy()
    # os.remove(temp_path)
    return result


def reload_outputs(node, node_name):
    tmp_image_node = node.node(node_name)
    if tmp_image_node is None:
        return

    try:
        tmp_image_node.parm("reload").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}: {str(e)}")


def get_node_parameters(node):
    """Extract all top-level parameters from the HDA."""
    if node is None:
        return {}

    params = {}
    for parm_tuple in node.parmTuples():
        values = [parm.eval() for parm in parm_tuple]
        params[parm_tuple.name()] = values[0] if len(values) == 1 else values
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
        image = input_to_base64(current, "src")
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

        print("get_ip_adapters", current, current.name())

        params = get_node_parameters(current)
        image = input_to_base64(current, "src")
        mask = input_to_base64(current, "mask")

        if image is None:
            continue

        tmp = IpAdapterModel(
            model=IpAdapterModelModel(params.get("model", "")),
            image=image,
            mask=mask,
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
