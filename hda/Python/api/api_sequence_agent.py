import json
import time

import hou

from config import client
from generated.api_client.api.agentic import sequence_agent
from generated.api_client.models.sequence_request import SequenceRequest
from generated.api_client.models.sequence_response import SequenceResponse
from generated.api_client.models.shot_response import ShotResponse
from utils import add_call_metadata, extract_and_format_parameters


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


def main(node):

    params = extract_and_format_parameters(node)
    valid_params = {k: v for k, v in params.items() if k in SequenceRequest.__annotations__}
    body = SequenceRequest(**valid_params)

    # make the API call
    start_time = time.time()
    response = sequence_agent.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, SequenceResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    # apply back to the node
    # chain_of_thought_str = json.dumps(response.parsed.chain_of_thought, indent=2)
    # response_str = split_text(str(response.parsed.response))

    # node.parm("chain_of_thought").set(chain_of_thought_str)
    node.parm("response").set(json.dumps(response.parsed.to_dict(), indent=2))
    add_call_metadata(node, body.to_dict(), response.parsed.to_dict(), start_time)

    for shot in response.parsed.shots:

        # Create a new subnet for each shot
        shot_node = node.parent().createNode("null", node_name=f"shot_{shot.name}")

        # Position it relative to the current node
        shot_node.setPosition(node.position())

        # Create parameters on the shot node
        parm_group = shot_node.parmTemplateGroup()
        shot_dict = shot.to_dict()

        for param_name, param_value in shot_dict.items():
            try:
                # Create string parameter with proper name and value
                parm_template = hou.StringParmTemplate(
                    name=param_name,  # Parameter name
                    label=param_name,  # .replace("_", " ").title(),  # Nice looking label
                    num_components=1,
                    default_value=[str(param_value)],  # Convert value to string
                )
                parm_group.addParmTemplate(parm_template)
            except Exception as e:
                print(f"Error adding parameter {param_name}: {e}")

        # Apply the parameter template
        shot_node.setParmTemplateGroup(parm_group)

        # Layout the nodes nicely
        shot_node.moveToGoodPosition()
