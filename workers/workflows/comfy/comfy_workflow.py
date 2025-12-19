import copy
import uuid
from typing import List

from PIL import Image

from common.logger import log_pretty, task_log
from common.memory import free_gpu_memory
from common.pipeline_helpers import clear_global_pipeline_cache
from workflows.comfy.comfy_client import (
    api_free,
    api_history,
    api_image_upload,
    api_prompt,
    api_view,
    is_comfy_running,
)
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowOutput, WorkflowRequest


def poll_until_resolved(prompt_id: str, timeout: int = 300, poll_interval: int = 1) -> dict:
    """Wait for the ComfyUI prompt to complete processing."""
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        history = api_history(prompt_id)
        log_pretty(f"History {prompt_id}", history)

        if prompt_id in history and "status" in history[prompt_id]:
            status = history[prompt_id]["status"]
            if status.get("status_str") in ["error", "failed", "cancelled", "failure"]:
                raise RuntimeError(f"ComfyUI workflow {prompt_id} encountered a problem: {status}")

        if prompt_id in history and "outputs" in history[prompt_id]:
            outputs = history[prompt_id]["outputs"]
            if outputs:
                return outputs
        time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI workflow did not complete within {timeout} seconds")


def patch_workflow(workflow_request: WorkflowRequest) -> dict:
    uuid_str = str(uuid.uuid4())  # could use task id if available
    remapped = copy.deepcopy(workflow_request.workflow)

    for patch in workflow_request.patches:
        target_node = None
        for _, node in remapped.items():
            if isinstance(node, dict) and node.get("_meta", {}).get("title") == patch.title:
                target_node = node
                break

        if target_node is None:
            raise ValueError(f"Patch title '{patch.title}' not found in workflow")

        inputs = target_node.get("inputs")
        if inputs is None:
            raise ValueError(f"Node '{patch.title}' has no inputs to patch")

        if patch.class_type == "LoadImage":
            uploaded_name = api_image_upload(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.png",
            )
            inputs["image"] = uploaded_name
        elif patch.class_type == "LoadVideo":
            uploaded_name = api_image_upload(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.mp4",
            )
            inputs["video"] = uploaded_name
        else:
            inputs["value"] = patch.value

    return remapped


def free_all() -> None:
    """Free all ComfyUI resources. Aggressive cleanup."""
    clear_global_pipeline_cache()
    free_gpu_memory()
    api_free(unload_models=True, free_memory=True)


def main(context: WorkflowContext) -> List[WorkflowOutput]:
    """Execute a ComfyUI workflow with optional patches and return the generated image."""
    if not is_comfy_running():
        raise RuntimeError("ComfyUI is not running")

    free_all()
    workflow = patch_workflow(context.data)
    log_pretty("Remapped ComfyUI workflow", workflow)

    # Queue the workflow to ComfyUI
    task_log("Queuing ComfyUI workflow...")
    queue_response = api_prompt(workflow)
    prompt_id = queue_response.get("prompt_id")
    if not prompt_id:
        raise ValueError("Failed to queue ComfyUI workflow")

    # Wait for the workflow to complete
    task_log(f"Waiting for ComfyUI workflow {prompt_id} to complete...")
    outputs = poll_until_resolved(prompt_id)
    free_all()

    # Find the first image in the outputs
    result: List[WorkflowOutput] = []
    for node_id, node_output in outputs.items():
        for output_name, output_data in node_output.items():
            if output_data and isinstance(output_data, list) and len(output_data) > 0:
                if "filename" in output_data[0] and "type" in output_data[0]:
                    if output_data[0]["type"] == "output":
                        task_log(f"{output_name} - {output_data[0]}")
                        filename = output_data[0]["filename"]
                        subfolder = output_data[0].get("subfolder", "")

                        if filename.endswith(".png"):
                            result.append(
                                WorkflowOutput(
                                    data_type="image",
                                    base64_data=api_view(filename, subfolder),
                                    filename=filename,
                                )
                            )
                        elif filename.endswith(".mp4"):
                            result.append(
                                WorkflowOutput(
                                    data_type="video",
                                    base64_data=api_view(filename, subfolder),
                                    filename=filename,
                                )
                            )

    if not result:
        raise ValueError("ComfyUI workflow did not produce any valid image or video outputs")

    return result
