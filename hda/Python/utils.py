import json
import time

import hou
from config import MAX_ADDITIONAL_IMAGES


def save_tmp_image(node, node_name):
    tmp_image_node = node.node(node_name)
    if tmp_image_node is None:
        return

    try:
        tmp_image_node.parm("execute").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def reload_outputs(node, node_name):
    tmp_image_node = node.node(node_name)
    if tmp_image_node is None:
        return

    try:
        tmp_image_node.parm("reload").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def extract_and_format_parameters(node):
    """Extract all top-level parameters from the HDA."""
    if node is None:
        return {}

    params = {}
    for parm_tuple in node.parmTuples():
        values = [parm.eval() for parm in parm_tuple]
        params[parm_tuple.name()] = values[0] if len(values) == 1 else values

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
        key_map[f"controlnet_{i}"] = f"controlnet_{i}_path"

    for key, param_key in key_map.items():
        if key not in valid_inputs and param_key in params:
            params.pop(param_key)

    return params


def get_control_nets(params):
    controlnets = []
    for i in range(MAX_ADDITIONAL_IMAGES):
        if f"controlnet_{i}_path" in params:
            tmp = {
                "model": params.get(f"controlnet_{i}_model", ""),
                "input_image": params.get(f"controlnet_{i}_path", ""),
                "conditioning_scale": params.get(f"controlnet_{i}_conditioning_scale", 0.5),
                "current": f"controlnet_{i}",
            }
            if tmp["model"] != "":
                controlnets.append(tmp)

    return controlnets


def add_call_metadata(node, body, response_content, start_time):
    call_metadata = {
        "body": body,
        "response": response_content,
        "time_taken": round(time.time() - start_time, 5),
    }
    call_metadata_str = json.dumps(call_metadata, indent=2)

    if node.parm("call_metadata"):
        node.parm("call_metadata").set(call_metadata_str)
