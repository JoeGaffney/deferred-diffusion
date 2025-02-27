import json
import os
import time

import hou
import requests

MAX_ADDITIONAL_IMAGES = 3


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

    # Extract the controlnets all parameters should be prefixed with 'controlnet_*'
    params["controlnets"] = get_control_nets(params)

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


def trigger_api(node, mode="image"):
    # Save the specific ROP node 'tmp_input_image'
    save_tmp_image(node, "tmp_input_image")
    save_tmp_image(node, "tmp_input_mask")
    for i in range(MAX_ADDITIONAL_IMAGES):
        save_tmp_image(node, f"tmp_controlnet_{i}")
        save_tmp_image(node, f"tmp_image_{i}")

    # Extract top-level parameters
    parameters = extract_and_format_parameters(node)
    # print(f"Extracted Parameters: {parameters}")

    # API Call
    api_root = os.getenv("DD_SERVER_ADDRESS", "http://127.0.0.1:5000/")
    api_url = f"{api_root}/api/{mode}"
    body = parameters

    response = None
    try:
        start_time = time.time()
        response = requests.post(api_url, json=body)
        if response.status_code != 200:
            hou.ui.displayMessage(f"API Call Failed: {response.text}")
            return

        print(f"API Response: {response.text}")
        reload_outputs(node, "output_read")
        reload_outputs(node, "output_read_video")
    except Exception as e:
        error_response = f"API Call Failed: {str(e)}"
        hou.ui.displayMessage(error_response)
        return


def api_image(kwargs=None):
    if kwargs is None:
        return

    node = kwargs.get("node")
    if node is None:
        hou.ui.displayMessage("Node not found in kwargs!")
        return

    trigger_api(node, "image")


def api_video(kwargs=None):
    if kwargs is None:
        return

    node = kwargs.get("node")
    if node is None:
        hou.ui.displayMessage("Node not found in kwargs!")
        return

    trigger_api(node, "video")


def api_text(kwargs=None):
    if kwargs is None:
        return

    node = kwargs.get("node")
    if node is None:
        hou.ui.displayMessage("Node not found in kwargs!")
        return

    trigger_api(node, "text")


def api_image_frame_range(kwargs=None):
    start_frame = int(hou.playbar.frameRange().x())
    end_frame = int(hou.playbar.frameRange().y())
    print(f"Frame Range: {start_frame} - {end_frame}")

    for frame in range(start_frame, end_frame + 1):
        hou.setFrame(frame)
        api_image(kwargs)
        # hou.ui.waitUntil(lambda: False)  # Allow Houdini to update the UI
