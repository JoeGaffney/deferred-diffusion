from PIL import Image

from common.logger import log_pretty
from workflows.comfy.utils import (
    ensure_comfy_alive,
    get_image,
    poll_until_resolved,
    queue_prompt,
    remap_workflow,
)
from workflows.context import WorkflowContext


def main(context: WorkflowContext) -> Image.Image:
    ensure_comfy_alive()

    workflow = remap_workflow(context.data.workflow_json, context.data.patches)
    log_pretty("Remapped ComfyUI workflow", workflow)

    # Queue the workflow to ComfyUI
    queue_response = queue_prompt(workflow)
    prompt_id = queue_response.get("prompt_id")
    if not prompt_id:
        raise ValueError("Failed to queue ComfyUI workflow")

    # Wait for the workflow to complete
    outputs = poll_until_resolved(prompt_id)

    # Find the first image in the outputs
    result = None
    for node_id, node_output in outputs.items():
        for output_name, output_data in node_output.items():
            if output_data and isinstance(output_data, list) and len(output_data) > 0:
                if "filename" in output_data[0] and "type" in output_data[0]:
                    if output_data[0]["type"] == "output":
                        # Get the generated image
                        filename = output_data[0]["filename"]
                        subfolder = output_data[0].get("subfolder", "")
                        result = get_image(filename, subfolder)
                        break
        if result:
            break

    if isinstance(result, Image.Image):
        return result

    raise ValueError("Image generation failed")
