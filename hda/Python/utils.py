import json
import os
import hou
import requests
import time

MAX_ADDITIONAL_IMAGES = 3


def split_text(text, max_length=120):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


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


def extract_and_format_parameters(hda_node):
    """Extract all top-level parameters from the HDA."""
    if hda_node is None:
        return {}

    params = {}
    for parm_tuple in hda_node.parmTuples():
        values = [parm.eval() for parm in parm_tuple]
        params[parm_tuple.name()] = values[0] if len(values) == 1 else values

    # Remove 'images' if not in valid_inputs
    valid_inputs = []
    for i in hda_node.inputConnections():
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

    # apply messages
    params["messages"] = get_messages(params)

    return params


def get_messages(params):

    # get the current message
    message_content = []

    for i in range(MAX_ADDITIONAL_IMAGES):
        if f"image_{i}_path" in params:
            message_content.append(
                {
                    "type": "image",
                    "image": params[f"image_{i}_path"],
                }
            )
    message_content.append({"type": "text", "text": params.get("prompt", "")})
    message = [
        {
            "role": "user",
            "content": message_content,
        }
    ]

    # factor previous chaing of thought
    previous_messages = []
    previous_messages_str = params.get("previous_messages", "[]")
    try:
        previous_messages = json.loads(previous_messages_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        previous_messages = []

    messages = previous_messages + message
    return messages


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


def add_response_data(node, body, mode, response, start_time):
    formatted_time = round(time.time() - start_time, 5)

    call_metadata = {
        "body": body,
        "mode": mode,
        "response": response.text,
        "time_taken": formatted_time,
    }

    # Convert dictionary to a neatly formatted string
    call_metadata_str = json.dumps(call_metadata, indent=2)  # More readable

    # Check if the parameter already exists
    if not node.parm("call_metadata"):
        parm_template = hou.StringParmTemplate(
            "call_metadata", "Call Metadata", 1, string_type=hou.stringParmType.Regular
        )
        parm_template.setTags({"editor": "1", "editorlang": "python"})
        node.addSpareParmTuple(parm_template)

    node.parm("call_metadata").set(call_metadata_str)

    # apply json response to the text node
    json_response = response.json() or {}

    # apply to the text node
    if json_response.get("chain_of_thought") and node.parm("chain_of_thought"):
        chain_of_thought_str = json.dumps(json_response.get("chain_of_thought"), indent=2)  # More readable
        node.parm("chain_of_thought").set(chain_of_thought_str)

    if json_response.get("response") and node.parm("response"):
        # text box does not wrap text
        node.parm("response").set(split_text(str(json_response["response"])))


def trigger_api(node, mode="image"):
    # Save the specific ROP node 'tmp_input_image'
    save_tmp_image(node, "tmp_input_image")
    save_tmp_image(node, "tmp_input_mask")
    for i in range(MAX_ADDITIONAL_IMAGES):
        save_tmp_image(node, f"tmp_controlnet_{i}")
        save_tmp_image(node, f"tmp_image_{i}")

    # Extract top-level parameters
    parameters = extract_and_format_parameters(node)
    print(f"Extracted Parameters: {parameters}")

    # API Call
    api_root = os.getenv("DD_SERVER_ADDRESS", "http://127.0.0.1:5000/api")
    api_url = f"{api_root}/{mode}"
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

    if response is not None:
        add_response_data(node, body, mode, response, start_time)


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
