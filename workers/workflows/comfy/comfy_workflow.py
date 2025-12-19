import copy
import uuid

from common.logger import log_pretty, task_log
from common.memory import free_gpu_memory
from common.pipeline_helpers import clear_global_pipeline_cache
from workflows.comfy.comfy_client import ComfyClient
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowOutput, WorkflowRequest


def patch_workflow(workflow_request: WorkflowRequest, comfy: ComfyClient) -> dict:
    uuid_str = str(uuid.uuid4())
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
            uploaded_name = comfy.upload_image(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.png",
            )
            inputs["image"] = uploaded_name
        elif patch.class_type == "LoadVideo":
            uploaded_name = comfy.upload_image(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.mp4",
            )
            inputs["video"] = uploaded_name
        else:
            inputs["value"] = patch.value

    return remapped


def free_all(comfy: ComfyClient) -> None:
    """Free all ComfyUI resources. Aggressive cleanup."""
    clear_global_pipeline_cache()
    free_gpu_memory()
    comfy.free_memory(unload_models=True, free_memory=True)


def main(context: WorkflowContext) -> List[WorkflowOutput]:
    """Execute a ComfyUI workflow with optional patches and return the generated image."""
    with ComfyClient() as comfy:
        if not comfy.is_running():
            raise RuntimeError("ComfyUI is not running")

        free_all(comfy)
        workflow = patch_workflow(context.data, comfy)
        log_pretty("Remapped ComfyUI workflow", workflow)

        queue_response = comfy.queue_prompt(workflow)
        prompt_id = queue_response.get("prompt_id")
        if not prompt_id:
            raise ValueError("Failed to queue ComfyUI workflow")

        completed = comfy.track_progress(prompt_id)
        if not completed:
            raise RuntimeError(f"ComfyUI workflow {prompt_id} did not complete successfully")

        outputs = comfy.get_completed_history(prompt_id)

        free_all(comfy)

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
                                        base64_data=comfy.download_file(filename, subfolder),
                                        filename=filename,
                                    )
                                )
                            elif filename.endswith(".mp4"):
                                result.append(
                                    WorkflowOutput(
                                        data_type="video",
                                        base64_data=comfy.download_file(filename, subfolder),
                                        filename=filename,
                                    )
                                )

        if not result:
            raise ValueError("ComfyUI workflow did not produce any valid image or video outputs")

        return result
