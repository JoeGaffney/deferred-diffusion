from PIL import Image

from images.context import ImageContext
from images.models.auto_diffusion import main as auto_diffusion
from images.models.auto_openai import main as auto_openai
from images.models.auto_runway import main as auto_runway
from images.models.comfy import main as comfy
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
    family = context.model_config.model_family

    result = None
    if family in ["sd1.5", "sdxl", "sd3", "hidream", "flux"]:
        result = auto_diffusion(context)
    elif family == "openai":
        result = auto_openai(context)
    elif family == "runway":
        result = auto_runway(context)
    elif family == "sd_upscaler":
        result = stable_diffusion_upscaler(context)
    elif family == "depth_anything":
        result = depth_anything(context)
    elif family == "segment_anything":
        result = segment_anything(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    if isinstance(result, Image.Image):
        # save a temp file for now
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")


@celery_app.task(name="process_image_workflow")
def process_image_workflow(request_dict):
    free_gpu_memory()

    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)

    result = comfy(context)

    if isinstance(result, Image.Image):
        # save a temp file for now
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")
