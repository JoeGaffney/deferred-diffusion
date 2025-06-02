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

    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)

    result = comfy(context)

    if isinstance(result, Image.Image):
        # save a temp file for now
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")
