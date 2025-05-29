import json

from PIL import Image

from common.comfy.comfy_utils import (
    ensure_comfy_alive,
    get_image,
    queue_prompt,
    wait_for_completion,
)
from images.context import ImageContext
from images.models.auto_diffusion import main as auto_diffusion
from images.models.auto_openai import main as auto_openai
from images.models.auto_runway import main as auto_runway
from images.models.depth_anything import main as depth_anything
from images.models.segment_anything import main as segment_anything
from images.models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from images.schemas import ImageRequest, ImageWorkerResponse
from utils.utils import free_gpu_memory, pil_to_base64
from worker import celery_app


@celery_app.task(name="process_image")
def process_image(request_dict):
    free_gpu_memory()

    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    mode = context.model_config.mode
    family = context.model_config.model_family

    if mode == "upscale":
        result = stable_diffusion_upscaler(context, mode=mode)
    elif mode == "depth":
        result = depth_anything(context, mode=mode)
    elif mode == "mask":
        result = segment_anything(context, mode=mode)
    else:
        # auto_diffusion
        auto_mode = "img_to_img"
        if context.data.mask:
            auto_mode = "img_to_img_inpainting"
        if context.data.image is None:
            auto_mode = "text_to_image"

        if family == "openai":
            result = auto_openai(context, mode=auto_mode)
        elif family == "runway":
            result = auto_runway(context, mode=auto_mode)
        else:
            result = auto_diffusion(context, mode=auto_mode)

    if isinstance(result, Image.Image):
        # save a temp file for now
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")


@celery_app.task(name="process_image_workflow")
def process_image_workflow(request_dict):
    free_gpu_memory()
    ensure_comfy_alive()

    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    comfy_workflow = context.data.comfy_workflow
    if not comfy_workflow:
        raise ValueError("Comfy workflow is required for this task")

    # Ensure workflow is a dictionary
    if isinstance(comfy_workflow, str):
        workflow = json.loads(comfy_workflow)
    else:
        workflow = comfy_workflow

    # Queue the workflow to ComfyUI
    queue_response = queue_prompt(workflow)
    prompt_id = queue_response.get("prompt_id")

    if not prompt_id:
        raise ValueError("Failed to queue ComfyUI workflow")

    # Wait for the workflow to complete
    outputs = wait_for_completion(prompt_id)

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
        # save a temp file for now
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")
