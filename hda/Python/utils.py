import hou
import requests


def save_tmp_image(node, node_name):
    tmp_image_node = node.node(node_name)

    if tmp_image_node is None:
        hou.ui.displayMessage("'f{node_name}' not found!")
        return

    try:
        tmp_image_node.parm("execute").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def reload_outputs(node, node_name):
    tmp_image_node = node.node(node_name)

    if tmp_image_node is None:
        hou.ui.displayMessage("'f{node_name}' not found!")
        return

    try:
        tmp_image_node.parm("reload").pressButton()  # Trigger execution
    except Exception as e:
        hou.ui.displayMessage(f"Failed to save '{tmp_image_node.name()}': {str(e)}")


def get_top_level_parameters(hda_node):
    """Extract all top-level parameters from the HDA."""
    if hda_node is None:
        return {}

    params = {}
    for parm_tuple in hda_node.parmTuples():
        values = [parm.eval() for parm in parm_tuple]
        params[parm_tuple.name()] = values[0] if len(values) == 1 else values
    return params


def trigger_api(kwargs=None, model_type="img_to_img"):
    """Triggers the API, saves the ROP node, and extracts parameters."""
    if kwargs is None:
        return

    node = kwargs.get("node")
    if node is None:
        hou.ui.displayMessage("Node not found in kwargs!")
        return

    # Save the specific ROP node 'tmp_input_image'
    if model_type == "video_to_img":
        save_tmp_image(node, "tmp_input_image")
    elif model_type == "img_to_img":
        save_tmp_image(node, "tmp_input_image")
        save_tmp_image(node, "tmp_input_mask")

    # Extract top-level parameters
    parameters = get_top_level_parameters(node)
    print(f"Extracted Parameters: {parameters}")
    # hou.ui.displayMessage(f"Extracted Parameters: {parameters}")

    # API Call
    api_root = "http://127.0.0.1:5000/api"
    api_url = f"{api_root}/{model_type}"
    body = parameters

    try:
        response = requests.post(api_url, json=body)
        # hou.ui.displayMessage(f"API Response: {response.text}")
        print(f"API Response: {response.text}")
        if model_type == "img_to_img":
            reload_outputs(node, "output_read")
        elif model_type == "text_to_img":
            reload_outputs(node, "output_read")
        elif model_type == "img_to_video":
            reload_outputs(node, "output_read_video")
    except Exception as e:
        hou.ui.displayMessage(f"API Call Failed: {str(e)}")


def api_img_to_img(kwargs=None):
    trigger_api(kwargs, "img_to_img")


def api_img_to_video(kwargs=None):
    trigger_api(kwargs, "img_to_video")


def api_text_to_img(kwargs=None):
    trigger_api(kwargs, "text_to_img")
