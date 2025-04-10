import json
import time

import hou

from config import client
from generated.api_client.api.agentic import sequence_agent
from generated.api_client.models.sequence_request import SequenceRequest
from generated.api_client.models.sequence_response import SequenceResponse
from generated.api_client.models.shot_response import ShotResponse
from utils import add_call_metadata, add_spare_params, extract_and_format_parameters


def create_image_node(node, node_name):
    nice_name = node_name.replace(" ", "_").replace(":", "_").replace("-", "_")
    print(f"Creating node: {nice_name}")
    result = node.parent().createNode("deferred_diffusion::image", node_name=nice_name)
    result.setPosition(node.position())
    result.moveToGoodPosition()
    return result


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

    # set the node parameters
    node.parm("response").set(json.dumps(response.parsed.to_dict(), indent=2))
    add_call_metadata(node, body.to_dict(), response.parsed.to_dict(), start_time)

    # build out the nodes
    scene = response.parsed.scene
    node_name = node.name()

    scene_node = create_image_node(node, node_name=f"{node_name}_scene_{scene.name}_image")
    add_spare_params(scene_node, "scene", scene.to_dict())
    scene_node.parm("prompt").set(f"{scene.image_description}, {scene.diffusion_postive_prompt_tags}")
    scene_node.parm("negative_prompt").set(scene.diffusion_negative_prompt_tags)
    scene_node.parm("max_width").set(1280)
    scene_node.parm("max_height").set(768)
    scene_node.parm("model").set("stabilityai/stable-diffusion-xl-base-1.0")

    for character in response.parsed.characters:
        character_node = create_image_node(node, node_name=f"{node_name}_character_{character.name}_image")

        add_spare_params(character_node, "scene", scene.to_dict())
        add_spare_params(character_node, "character", character.to_dict())
        character_node.parm("prompt").set(f"{character.image_description}, {scene.diffusion_postive_prompt_tags}")
        character_node.parm("negative_prompt").set(scene.diffusion_negative_prompt_tags)
        character_node.parm("max_width").set(512)
        character_node.parm("max_height").set(512)
        character_node.parm("model").set("stabilityai/stable-diffusion-xl-base-1.0")

    for shot in response.parsed.shots:
        shot_node = create_image_node(node, node_name=f"{node_name}_shot_{shot.name}_image")

        add_spare_params(shot_node, "scene", scene.to_dict())
        add_spare_params(shot_node, "shot", shot.to_dict())
        shot_node.parm("prompt").set(f"{shot.image_description}, {scene.diffusion_postive_prompt_tags}")
        shot_node.parm("negative_prompt").set(scene.diffusion_negative_prompt_tags)
        shot_node.parm("max_width").set(1280)
        shot_node.parm("max_height").set(768)
        shot_node.parm("model").set("stabilityai/stable-diffusion-xl-base-1.0")
