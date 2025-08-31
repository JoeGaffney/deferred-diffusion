from PIL import Image

from images.context import ImageContext

# model handler imports are moved to inside each task to avoid heavy import-time work
from images.schemas import EXTERNAL_MODELS, ImageRequest, ImageWorkerResponse
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


def router_main(request: ImageRequest):
    """Enqueue the explicit per-model celery task using the ImageRequest.

    Calls the celery task registered under the user-facing model name and
    returns the AsyncResult.
    """
    # model names are registered as celery task names (e.g. "flux-1", "sd-xl")
    task_name = request.model
    queue = request.task_queue

    # serialize the ImageRequest for the task
    task = celery_app.signature(task_name, args=(request.model_dump(),), queue=queue)
    async_result = task.apply_async()
    return async_result


# Explicit internal model tasks (lazy-import model implementation inside each task)
@celery_app.task(name="sd-3", queue="gpu")
def sd_3(request_dict):
    from images.models.sd3 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="flux-1", queue="gpu")
def flux_1(request_dict):
    from images.models.flux import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="flux-1-krea", queue="gpu")
def flux_1_krea(request_dict):
    from images.models.flux import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="flux-kontext-1", queue="gpu")
def flux_kontext_1(request_dict):
    from images.models.flux_kontext import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="qwen-image", queue="gpu")
def qwen_image(request_dict):
    from images.models.qwen import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="depth-anything-2", queue="gpu")
def depth_anything_2(request_dict):
    from images.models.depth_anything import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="segment-anything-2", queue="gpu")
def segment_anything_2(request_dict):
    from images.models.segment_anything import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="real-esrgan-x4", queue="gpu")
def real_esrgan_x4(request_dict):
    from images.models.real_esrgan import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="sd-xl", queue="gpu")
def sd_xl(request_dict):
    from images.models.sdxl import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@celery_app.task(name="gpt-image-1", queue="cpu")
def gpt_image_1(request_dict):
    from images.external_models.openai import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="runway-gen4-image", queue="cpu")
def runway_gen4_image(request_dict):
    from images.external_models.runway import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="flux-kontext-1-pro", queue="cpu")
def flux_kontext_1_pro(request_dict):
    from images.external_models.flux_kontext import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="flux-1-1-pro", queue="cpu")
def flux_1_1_pro(request_dict):
    from images.external_models.flux import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="topazlabs-upscale", queue="cpu")
def topazlabs_upscale(request_dict):
    from images.external_models.topazlabs import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
