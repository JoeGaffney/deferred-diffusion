import json

import hou

from config import client
from generated.api_client.api.agentic import agentic_sequence_create
from generated.api_client.models.character_response import CharacterResponse
from generated.api_client.models.scene_response import SceneResponse
from generated.api_client.models.sequence_request import SequenceRequest
from generated.api_client.models.sequence_response import SequenceResponse
from utils import add_spare_params, get_node_parameters, input_to_base64


def create_image_node(node, node_name):
    nice_name = node_name.replace(" ", "_").replace(":", "_").replace("-", "_")
    result = node.parent().createNode("deferred_diffusion::image", node_name=nice_name)
    result.setPosition(node.position())
    result.moveToGoodPosition()
    return result


def create_shot_node(node, node_name):
    nice_name = node_name.replace(" ", "_").replace(":", "_").replace("-", "_")
    result = node.parent().createNode("deferred_diffusion::shot", node_name=nice_name)
    result.setPosition(node.position())
    result.moveToGoodPosition()
    return result


def create_character_node(node, scene: SceneResponse, character: CharacterResponse):
    node_name = f"{node.name()}_character_{character.name}"
    node_name = node_name.replace(" ", "_").replace(":", "_").replace("-", "_")

    result = node.parent().createNode("deferred_diffusion::image", node_name=node_name)
    result.setPosition(node.position())
    result.moveToGoodPosition()

    add_spare_params(result, "scene", scene.to_dict())
    add_spare_params(result, "character", character.to_dict())
    result.parm("prompt").set(f"{character.image_prompt}, {scene.diffusion_positive_prompt_tags}")
    result.parm("width").set(1024)
    result.parm("height").set(1024)
    return result


def main(node):
    params = get_node_parameters(node)
    body = SequenceRequest(
        prompt=params.get("prompt", ""),
        refinement_prompt=params.get("refinement_prompt", ""),
        scene_reference_image=input_to_base64(node, "scene"),
        protagonist_reference_image=input_to_base64(node, "protagonist"),
        antagonist_reference_image=input_to_base64(node, "antagonist"),
    )

    # make the API call
    response = agentic_sequence_create.sync_detailed(client=client, body=body)
    if response.status_code != 200:
        hou.ui.displayMessage(f"API Call Failed: {response}")
        return

    if not isinstance(response.parsed, SequenceResponse):
        hou.ui.displayMessage(f"Invalid response type: {type(response.parsed)} {response}")
        return

    # set the node parameters
    node.parm("response").set(json.dumps(response.parsed.to_dict(), indent=2))

    # prep generated nodes
    node_name = node.name()
    scene = response.parsed.scene

    # build the scene node
    scene_node = create_image_node(node, node_name=f"{node_name}_scene_{scene.name}")
    add_spare_params(scene_node, "scene", scene.to_dict())
    scene_node.parm("prompt").set(f"{scene.image_prompt}, {scene.diffusion_positive_prompt_tags}")
    scene_node.parm("width").set(1280)
    scene_node.parm("height").set(768)

    # build the character nodes
    protagonist_node = None
    if response.parsed.protagonist:
        protagonist_node = create_character_node(node, scene, response.parsed.protagonist)

    antagonist_node = None
    if response.parsed.antagonist:
        antagonist_node = create_character_node(node, scene, response.parsed.antagonist)

    # build the shot nodes
    for shot in response.parsed.shots:
        shot_node = create_shot_node(node, node_name=f"{node_name}_shot_{shot.name}")
        add_spare_params(shot_node, "scene", scene.to_dict())
        add_spare_params(shot_node, "shot", shot.to_dict())
        shot_node.parm("prompt").set(f"{shot.image_prompt}, {scene.diffusion_positive_prompt_tags}")
        shot_node.parm("video_prompt").set(f"{shot.video_prompt}")
        shot_node.setInput(0, scene_node, 0)  # 0 is the "scene" input index on shot_node

        if protagonist_node and shot.protagonist:
            add_spare_params(shot_node, "protagonist", shot.protagonist.to_dict())
            shot_node.setInput(1, protagonist_node, 0)  # 1 is the "protagonist" input index on shot_node

        if shot.antagonist is not None:
            add_spare_params(shot_node, "antagonist", shot.antagonist.to_dict())
            shot_node.setInput(2, antagonist_node, 0)  # 2 is the "antagonist" input index on shot_node
