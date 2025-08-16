from PIL import Image

from images.context import ImageContext
from images.external_models.flux import main as external_flux_main
from images.external_models.flux_kontext import main as external_flux_kontext_main
from images.external_models.openai import main as external_openai_main
from images.external_models.runway import main as external_runway_main
from images.external_models.topazlabs import main as external_topazlabs_main
from images.models.depth_anything import main as depth_anything_main
from images.models.flux import main as flux_main
from images.models.flux_kontext import main as flux_kontext_main
from images.models.real_esrgan import main as real_esrgan_main
from images.models.sd3 import main as sd3_main
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


def model_router_main(context: ImageContext):
    family = context.data.model_family
    if family == "sdxl":
        return sdxl_main(context)
    elif family == "sd3":
        return sd3_main(context)
    elif family == "flux":
        return flux_main(context)
    elif family == "flux_kontext":
        return flux_kontext_main(context)
    elif family == "real_esrgan":
        return real_esrgan_main(context)
    elif family == "depth_anything":
        return depth_anything_main(context)
    elif family == "segment_anything":
        return segment_anything_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")


def external_model_router_main(context: ImageContext):
    family = context.data.model_family
    if family == "openai":
        return external_openai_main(context)
    elif family == "runway":
        return external_runway_main(context)
    elif family == "flux_kontext":
        return external_flux_kontext_main(context)
    elif family == "flux":
        return external_flux_main(context)
    elif family == "topazlabs":
        return external_topazlabs_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")


def router_main(context: ImageContext):
    if context.data.external_model:
        return external_model_router_main(context)

    return model_router_main(context)


@celery_app.task(name="process_image", queue="gpu")
def process_image(request_dict):
    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    result = model_router_main(context)

    return process_result(context, result)


@celery_app.task(name="process_image_external", queue="cpu")
def process_image_external(request_dict):
    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    result = external_model_router_main(context)

    return process_result(context, result)
