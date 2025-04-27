import base64
import os
from typing import Optional

import hou

from config import MAX_ADDITIONAL_IMAGES
from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.control_net_schema_model import ControlNetSchemaModel
from generated.api_client.models.ip_adapter_model import IpAdapterModel
from generated.api_client.models.ip_adapter_model_model import IpAdapterModelModel


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
