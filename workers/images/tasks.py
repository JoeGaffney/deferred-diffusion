from PIL import Image

from images.context import ImageContext
from images.schemas import ImageRequest, ImageWorkerResponse, ModelName
from utils.utils import pil_to_base64
from worker import celery_app


def process_result(context, result):
    if isinstance(result, Image.Image):
        context.save_image(result)
        return ImageWorkerResponse(base64_data=pil_to_base64(result)).model_dump()
    raise ValueError("Image generation failed")


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = ImageRequest.model_validate(request_dict)
    context = ImageContext(request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=name, queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="sd-3", queue="gpu")
def sd_3(request_dict):
    from images.local.sd_3 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1", queue="gpu")
def flux_1(request_dict):
    from images.local.flux_1 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="qwen-image", queue="gpu")
def qwen_image(request_dict):
    from images.local.qwen_image import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="depth-anything-2", queue="gpu")
def depth_anything_2(request_dict):
    from images.local.depth_anything_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="segment-anything-2", queue="gpu")
def segment_anything_2(request_dict):
    from images.local.segment_anything_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="real-esrgan-x4", queue="gpu")
def real_esrgan_x4(request_dict):
    from images.local.real_esrgan_x4 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sd-xl", queue="gpu")
def sd_xl(request_dict):
    from images.local.sd_xl import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="gpt-image-1", queue="cpu")
def gpt_image_1(request_dict):
    from images.external.gpt_image_1 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-gen4-image", queue="cpu")
def runway_gen4_image(request_dict):
    from images.external.runway_gen4_image import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1-pro", queue="cpu")
def flux_1_pro(request_dict):
    from images.external.flux_1_pro import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="topazlabs-upscale", queue="cpu")
def topazlabs_upscale(request_dict):
    from images.external.topazlabs_upscale import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="google-gemini-2", queue="cpu")
def gemini_2(request_dict):
    from images.external.google_gemini_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="bytedance-seedream-4", queue="cpu")
def seedream_4(request_dict):
    from images.external.bytedance_seedream_4 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
