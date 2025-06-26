from PIL import Image

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.external_models.flux_kontext import main as flux_kontext_main
from images.external_models.openai import main as openai_main
from images.external_models.runway import main as runway_main
from images.models.depth_anything import main as depth_anything_main
from images.models.flux import main as flux_main
from images.models.hidream import main as hidream_main
from images.models.sd3 import main as sd3_main
from images.models.sd_upscaler import main as sd_upscaler_main
from images.models.sdxl import main as sdxl_main
from images.models.segment_anything import main as segment_anything_main
from images.schemas import ImageRequest, ImageWorkerResponse
from utils.utils import pil_to_base64
from worker import celery_app


def process_result(context, result):
    if isinstance(result, Image.Image):
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()
    raise ValueError("Image generation failed")


@celery_app.task(name="process_image")
def process_image(request_dict):
    free_gpu_memory()
    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    family = context.data.model_family

    result = None
    if family == "sdxl":
        result = sdxl_main(context)
    elif family == "sd3":
        result = sd3_main(context)
    elif family == "flux":
        result = flux_main(context)
    elif family == "hidream":
        result = hidream_main(context)
    elif family == "sd_upscaler":
        result = sd_upscaler_main(context)
    elif family == "depth_anything":
        result = depth_anything_main(context)
    elif family == "segment_anything":
        result = segment_anything_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    return process_result(context, result)


@celery_app.task(name="process_image_external")
def process_image_external(request_dict):
    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    family = context.data.model_family

    result = None
    if family == "openai":
        result = openai_main(context)
    elif family == "runway":
        result = runway_main(context)
    elif family == "flux_kontext":
        result = flux_kontext_main(context)
    elif family == "flux_pro":
        # Assuming flux_pro is handled by the same function as flux_kontext
        result = flux_kontext_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    return process_result(context, result)
