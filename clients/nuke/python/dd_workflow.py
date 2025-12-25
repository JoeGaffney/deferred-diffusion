import base64
import json
import os
import time

import nuke
from httpx import RemoteProtocolError

from config import client
from generated.api_client.api.workflows import workflows_create, workflows_get
from generated.api_client.models import (
    Patch,
    PatchClassType,
    TaskStatus,
    WorkflowCreateResponse,
    WorkflowOutputDataType,
    WorkflowRequest,
    WorkflowRequestWorkflow,
    WorkflowResponse,
)
from utils import (
    COMPLETED_STATUS,
    base64_to_file,
    get_node_root_path,
    get_node_value,
    node_to_base64,
    node_to_base64_video,
    nuke_error_handling,
    polling_message,
    set_node_info,
    set_node_value,
    threaded,
    update_read_range,
)


def create_dd_workflow_node():
    node = nuke.createNode("dd_workflow")
    return node


def _insert_knobs_after(node, knob_name_ref, knob_list):
    """Insert a list of knobs after a given reference knob.

    Args:
        node: The node where the knobs will be added.
        knob_name_ref: The name of the reference knob. New knobs will be inserted after it.
        knob_list: List of knobs to add.
    """
    if not knob_list:
        return

    knob_backup_list = []
    make_backup = False
    active_tab = None

    # Save the knobs that are after the reference knob
    for k in node.allKnobs():
        if not make_backup and k.Class() == "Tab_Knob":
            active_tab = k
        if make_backup:
            knob_backup_list.append(k)
        elif k.name() == knob_name_ref:
            make_backup = True

    # Remove the knobs
    for k in knob_backup_list:
        node.removeKnob(k)

    # Add the new knobs
    for new_knob in knob_list:
        node.addKnob(new_knob)

    # Restore the backup of the removed knobs
    for k in knob_backup_list:
        node.addKnob(k)

    # Restore the active tab to prevent UI from jumping to wrong location
    if active_tab:
        active_tab.setFlag(0)


def _clear_dynamic_knobs(node):
    # Remove all knobs starting with p_
    # We iterate backwards to avoid index shifting issues during removal
    for i in range(node.numKnobs() - 1, -1, -1):
        k = node.knob(i)
        if k.name().startswith("p_"):
            node.removeKnob(k)

    # Clear internal Input nodes
    with node:
        for i in nuke.allNodes("Input"):
            nuke.delete(i)


def refresh_knobs(node):
    workflow_path = node["workflow_file"].value()

    # If path is empty, just clear the dynamic knobs and return
    if not workflow_path or not os.path.exists(workflow_path):
        _clear_dynamic_knobs(node)
        node["patch_info"].setValue("")
        return

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
    except Exception as e:
        nuke.message(f"Failed to load workflow: {str(e)}")
        return

    # Clear existing dynamic UI
    _clear_dynamic_knobs(node)

    # Don't store the full workflow JSON in the node - it can be large and cause encoding issues
    # We'll reload it from the file when needed

    # Regenerate Input nodes inside the group
    input_titles = []
    patch_info = {}

    for node_id, node_info in workflow_data.items():
        meta = node_info.get("_meta", {})
        title = meta.get("title")
        class_type = node_info.get("class_type")

        if not title or not class_type:
            continue

        # Only process nodes with supported PatchClassType values
        supported_types = [
            PatchClassType.LOADIMAGE,
            PatchClassType.LOADVIDEO,
            PatchClassType.PRIMITIVEINT,
            PatchClassType.PRIMITIVEFLOAT,
            PatchClassType.PRIMITIVESTRINGMULTILINE,
        ]
        if class_type not in supported_types:
            continue

        if class_type in [PatchClassType.LOADIMAGE, PatchClassType.LOADVIDEO]:
            input_titles.append(title)

        # Map knob name to original title and class_type
        # Nuke replaces spaces with underscores in knob names
        knob_name = f"p_{title.replace(' ', '_')}"
        patch_info[knob_name] = {"title": title, "class_type": class_type, "node_id": node_id}

    # Store patch mapping in hidden knob using base64 to avoid encoding issues
    patch_info_json = json.dumps(patch_info, ensure_ascii=False)
    patch_info_b64 = base64.b64encode(patch_info_json.encode("utf-8")).decode("ascii")
    node["patch_info"].setValue(patch_info_b64)

    with node:
        # Create new ones
        for i, title in enumerate(input_titles):
            # Sanitize title for node name
            safe_name = title.replace(" ", "_").replace("-", "_").replace(".", "_")
            new_input = nuke.nodes.Input(name=safe_name)
            # Position them
            new_input.setXYpos(70 + (i * 150), -200)

    # Collect new knobs to add inside the Parameters group
    new_knobs_to_add = []
    for knob_name, info in patch_info.items():
        title = info["title"]
        class_type = info["class_type"]
        node_info = workflow_data[info["node_id"]]
        label = title

        # Check if knob already exists
        if node.knob(knob_name):
            continue

        new_knob = None
        # Support both specific types and generic PrimitiveNode
        if class_type == PatchClassType.PRIMITIVEINT:
            new_knob = nuke.Int_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", 0)
            new_knob.setValue(int(val))
        elif class_type == PatchClassType.PRIMITIVEFLOAT:
            new_knob = nuke.Double_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", 0.0)
            new_knob.setValue(float(val))
        elif class_type == PatchClassType.PRIMITIVESTRINGMULTILINE or class_type == "PrimitiveString":
            new_knob = nuke.String_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", "")
            new_knob.setValue(str(val))
        elif class_type in [PatchClassType.LOADIMAGE, PatchClassType.LOADVIDEO]:
            # For loaders, we'll add a string knob to show which input index it's mapped to
            new_knob = nuke.String_Knob(knob_name, f"{label} (Input Index)")
            new_knob.setTooltip("Automatically mapped to Nuke input.")

            # Find the index in input_titles
            if title in input_titles:
                new_knob.setValue(str(input_titles.index(title)))
            new_knob.setEnabled(False)

        if new_knob:
            new_knobs_to_add.append(new_knob)

    # Insert new knobs after the Parameters knob (inside the Parameters group, before endGroup)
    if new_knobs_to_add:
        _insert_knobs_after(node, "Parameters", new_knobs_to_add)


def knob_changed():
    node = nuke.thisNode()
    knob = nuke.thisKnob()
    if knob.name() == "workflow_file":
        refresh_knobs(node)


@threaded
def _api_get_call(node, id, output_path: str, current_frame: int, iterations=300, sleep_time=5):
    set_node_info(node, TaskStatus.PENDING, "")

    for count in range(1, iterations + 1):
        time.sleep(sleep_time)

        try:
            parsed = workflows_get.sync(id, client=client)
            if not isinstance(parsed, WorkflowResponse):
                break
            if parsed.status in COMPLETED_STATUS:
                break

            def progress_update(parsed=parsed, count=count):
                set_node_info(node, parsed.status, polling_message(count, iterations, sleep_time), logs=parsed.logs)

            nuke.executeInMainThread(progress_update)
        except RemoteProtocolError:
            continue
        except Exception as e:

            def handle_error(error=e):
                with nuke_error_handling(node):
                    raise RuntimeError(f"API call failed: {str(error)}") from error

            nuke.executeInMainThread(handle_error)
            return

    def update_ui():
        with nuke_error_handling(node):
            if not isinstance(parsed, WorkflowResponse):
                raise ValueError("Unexpected response type from API call.")

            if not parsed.status == TaskStatus.SUCCESS or not parsed.result:
                raise ValueError(f"Task {parsed.status} with error: {parsed.error_message}")

            if not parsed.result.outputs:
                raise ValueError("No outputs found in workflow result.")

            # Position for new Read nodes
            spacing = 50
            start_x = node.xpos()
            start_y = node.ypos() + spacing
            root_path = get_node_root_path(node)

            for i, output in enumerate(parsed.result.outputs):
                data = output.base64_data
                ext = "png"
                if output.data_type == WorkflowOutputDataType.VIDEO:
                    ext = "mp4"

                # Generate unique path for each output
                final_output_path = f"{root_path}/{id}_{i}.{ext}"
                base64_to_file(data, final_output_path)

                # Create a new Read node outside the group
                read_name = f"{node.name()}_{i}"
                read_node = nuke.nodes.Read(name=read_name, file=final_output_path)
                read_node.setXYpos(start_x + (i * spacing), start_y)

                update_read_range(read_node)

            set_node_info(node, TaskStatus.SUCCESS, "", logs=parsed.logs)

    nuke.executeInMainThread(update_ui)


def process_workflow(node):
    set_node_info(node, None, "")
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        workflow_path = node["workflow_file"].value()
        patch_info_str = node["patch_info"].value()

        if not workflow_path or not os.path.exists(workflow_path):
            raise ValueError("Workflow file path is missing or invalid")

        if not patch_info_str:
            raise ValueError("Workflow data is missing")

        # Load workflow JSON from file instead of storing it in the node
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow_json = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load workflow file: {str(e)}") from e

        # Decode patch_info from base64 to handle special characters
        try:
            patch_info_json = base64.b64decode(patch_info_str.encode("ascii")).decode("utf-8")
            patch_info = json.loads(patch_info_json)
        except Exception as e:
            raise ValueError(f"Failed to decode patch info: {str(e)}") from e

        output_path = get_node_root_path(node)

        patches = []

        # Collect patches from dynamic knobs
        for knob_name, info in patch_info.items():
            k = node.knob(knob_name)
            if not k:
                continue

            title = info["title"]
            class_type = info["class_type"]
            value = k.value()

            # Special handling for LoadImage/LoadVideo
            if class_type in [PatchClassType.LOADIMAGE, PatchClassType.LOADVIDEO]:
                try:
                    # Value is the input index
                    input_idx = int(value)
                    input_node = node.input(input_idx)
                    if input_node:
                        if class_type == PatchClassType.LOADIMAGE:
                            value = node_to_base64(input_node, current_frame)
                        else:
                            # Use the explicit number_of_frames knob
                            num_frames = int(node["number_of_frames"].value())
                            value = node_to_base64_video(input_node, current_frame, num_frames=num_frames)
                    else:
                        continue  # Skip if no input connected
                except Exception as e:
                    nuke.tprint(f"Error patching {title}: {str(e)}")
                    continue
            elif class_type == PatchClassType.PRIMITIVEINT:
                value = int(value)

            patches.append(Patch(title=title, class_type=PatchClassType(class_type), value=value))

        workflow_req = WorkflowRequest(workflow=WorkflowRequestWorkflow.from_dict(workflow_json), patches=patches)

        try:
            parsed = workflows_create.sync(client=client, body=workflow_req)
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}") from e

        if not isinstance(parsed, WorkflowCreateResponse):
            raise ValueError(str(parsed))

        set_node_value(node, "task_id", str(parsed.id))
        _api_get_call(node, str(parsed.id), output_path, current_frame)


def get_workflow(node):
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        task_id = get_node_value(node, "task_id", "", mode="get")
        if not task_id or task_id == "":
            raise ValueError("Task ID is required to get the workflow result.")

        output_path = get_node_root_path(node)
        _api_get_call(node, task_id, output_path, current_frame, iterations=1, sleep_time=0)
