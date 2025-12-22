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
    WorkflowRequestWorkflowAdditionalProperty,
    WorkflowResponse,
)
from utils import (
    COMPLETED_STATUS,
    base64_to_file,
    get_node_value,
    get_output_path,
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


def refresh_knobs(node):
    workflow_path = node["workflow_file"].value()
    if not workflow_path or not os.path.exists(workflow_path):
        return

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
            # Store JSON in hidden knob to avoid re-reading file
            node["workflow_json"].setValue(json.dumps(workflow_data))
    except Exception as e:
        nuke.message(f"Failed to load workflow: {str(e)}")
        return

    # Find the Parameters group
    start_knob = node.knob("Parameters")
    end_knob = node.knob("endGroup")

    # Collect knobs to remove
    knobs_to_remove = []
    found_start = False
    for i in range(node.numKnobs()):
        k = node.knob(i)
        if k == start_knob:
            found_start = True
            continue
        if k == end_knob:
            break
        if found_start:
            knobs_to_remove.append(k)

    # Remove old knobs
    for k in knobs_to_remove:
        node.removeKnob(k)

    # Regenerate Input nodes inside the group
    input_titles = []
    patch_info = {}

    for node_id, node_info in workflow_data.items():
        meta = node_info.get("_meta", {})
        title = meta.get("title")
        class_type = node_info.get("class_type")

        if not title or not class_type:
            continue

        if class_type in ["LoadImage", "LoadVideo"]:
            input_titles.append(title)

        # Map knob name to original title and class_type
        # Nuke replaces spaces with underscores in knob names
        knob_name = f"p_{title.replace(' ', '_')}"
        patch_info[knob_name] = {"title": title, "class_type": class_type}

    # Store patch mapping in hidden knob
    node["patch_info"].setValue(json.dumps(patch_info))

    with node:
        # Find all existing Input nodes
        existing_inputs = nuke.allNodes("Input")
        for i in existing_inputs:
            nuke.delete(i)

        # Create new ones
        for i, title in enumerate(input_titles):
            # Sanitize title for node name
            safe_name = title.replace(" ", "_").replace("-", "_").replace(".", "_")
            new_input = nuke.nodes.Input(name=safe_name)
            # Position them
            new_input.setXYpos(70 + (i * 150), -200)

    # Add new knobs based on workflow
    # We look for nodes that have a title in _meta
    for node_id, node_info in workflow_data.items():
        meta = node_info.get("_meta", {})
        title = meta.get("title")
        class_type = node_info.get("class_type")

        if not title or not class_type:
            continue

        knob_name = f"p_{title.replace(' ', '_')}"
        label = title

        # Check if knob already exists (shouldn't if we removed them, but just in case)
        if node.knob(knob_name):
            continue

        new_knob = None
        if class_type == "PrimitiveInt":
            new_knob = nuke.Int_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", 0)
            new_knob.setValue(int(val))
        elif class_type == "PrimitiveFloat":
            new_knob = nuke.Double_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", 0.0)
            new_knob.setValue(float(val))
        elif class_type == "PrimitiveStringMultiline":
            new_knob = nuke.String_Knob(knob_name, label)
            val = node_info.get("inputs", {}).get("value", "")
            new_knob.setValue(str(val))
        elif class_type in ["LoadImage", "LoadVideo"]:
            # For loaders, we'll add a string knob to show which input index it's mapped to
            new_knob = nuke.String_Knob(knob_name, f"{label} (Input Index)")
            new_knob.setTooltip("Automatically mapped to Nuke input.")

            # Find the index in input_titles
            if title in input_titles:
                new_knob.setValue(str(input_titles.index(title)))
            new_knob.setEnabled(False)
        if new_knob:
            node.addKnob(new_knob)
            # Move it before the endGroup
            node.removeKnob(end_knob)
            node.addKnob(end_knob)


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
            start_x = node.xpos()
            start_y = node.ypos() + 100

            for i, output in enumerate(parsed.result.outputs):
                data = output.base64_data
                ext = "png"
                if output.data_type == WorkflowOutputDataType.VIDEO:
                    ext = "mp4"

                # Generate unique path for each output
                base_path = os.path.splitext(output_path)[0]
                final_output_path = f"{base_path}_{i}.{ext}"

                base64_to_file(data, final_output_path)

                # Create a new Read node outside the group
                safe_id = id.replace("-", "_")
                read_name = f"{node.name()}_{safe_id}_{i}"
                read_node = nuke.nodes.Read(name=read_name, file=final_output_path)
                read_node.setXYpos(start_x + (i * 100), start_y)

                update_read_range(read_node)

            set_node_info(node, TaskStatus.SUCCESS, "", logs=parsed.logs)

    nuke.executeInMainThread(update_ui)


def process_workflow(node):
    set_node_info(node, None, "")
    current_frame = nuke.frame()

    with nuke_error_handling(node):
        workflow_json_str = node["workflow_json"].value()
        patch_info_str = node["patch_info"].value()

        if not workflow_json_str or not patch_info_str:
            raise ValueError("Workflow data is missing. Please refresh parameters.")

        workflow_json = json.loads(workflow_json_str)
        patch_info = json.loads(patch_info_str)

        output_path = get_output_path(node, movie=False)  # Default to image path

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
            if class_type in ["LoadImage", "LoadVideo"]:
                try:
                    # Value is the input index
                    input_idx = int(value)
                    input_node = node.input(input_idx)
                    if input_node:
                        if class_type == "LoadImage":
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

        output_path = get_output_path(node, movie=False)
        _api_get_call(node, task_id, output_path, current_frame, iterations=1, sleep_time=0)
