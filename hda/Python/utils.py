import json
import time

import hou
from config import MAX_ADDITIONAL_IMAGES
from generated.api_client.models.control_net_schema import ControlNetSchema
from generated.api_client.models.ip_adapter_model import IpAdapterModel


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

        tmp = ControlNetSchema(
            model=params.get("model", ""),
            image_path=params.get("image_path", ""),
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

        tmp = IpAdapterModel(
            model=params.get("model", ""),
            image_path=params.get("image_path", ""),
            image_encoder=bool(params.get("image_encoder", False)),
            subfolder=params.get("subfolder", "models"),
            weight_name=params.get("weight_name", "ip-adapter_sd15.bin"),
            scale=params.get("scale", 0.5),
        )
        result.append(tmp)

    return result


def add_call_metadata(node, body, response_content, start_time):
    call_metadata = {
        "body": body,
        "response": response_content,
        "time_taken": round(time.time() - start_time, 5),
    }
    call_metadata_str = json.dumps(call_metadata, indent=2)

    if node.parm("call_metadata"):
        node.parm("call_metadata").set(call_metadata_str)
