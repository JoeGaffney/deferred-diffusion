from pathlib import Path
from typing import List

from PIL import Image

from common.config import settings
from common.logger import get_task_logs
from images.context import ImageContext
from images.schemas import ImageRequest, ImageWorkerResponse, ModelName
from worker import celery_app


def process_result(context: ImageContext, result: List[Path]):
    storage_dir = Path(settings.storage_dir)
    output = [str(path.relative_to(storage_dir)) for path in result]
    return ImageWorkerResponse(output=output, logs=get_task_logs()).model_dump()


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(args, **kwargs):
    request = ImageRequest.model_validate(args)
    context = ImageContext(request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=f"images.{name}", queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="sd-xl", queue="gpu")
def sd_xl(args, **kwargs):
    from images.local.sd_xl import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1", queue="gpu")
def flux_1(args, **kwargs):
    from images.local.flux_1 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-2", queue="gpu")
def flux_2(args, **kwargs):
    from images.local.flux_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="qwen-image", queue="gpu")
def qwen_image(args, **kwargs):
    from images.local.qwen_image import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="z-image", queue="gpu")
def z_image(args, **kwargs):
    from images.local.z_image import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="depth-anything-2", queue="gpu")
def depth_anything_2(args, **kwargs):
    from images.local.depth_anything_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sam-2", queue="gpu")
def sam_2_image(args, **kwargs):
    from images.local.sam_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sam-3", queue="gpu")
def sam_3_image(args, **kwargs):
    from images.local.sam_3 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="real-esrgan-x4", queue="gpu")
def real_esrgan_x4(args, **kwargs):
    from images.local.real_esrgan_x4 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="gpt-image-1", queue="cpu")
def gpt_image_1(args, **kwargs):
    from images.external.gpt_image_1 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-gen-4", queue="cpu")
def runway_gen4_image(args, **kwargs):
    from images.external.runway_gen_4 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1-pro", queue="cpu")
def flux_1_pro(args, **kwargs):
    from images.external.flux_1_pro import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-2-pro", queue="cpu")
def flux_2_pro(args, **kwargs):
    from images.external.flux_2_pro import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="topazlabs-upscale", queue="cpu")
def topazlabs_upscale(args, **kwargs):
    from images.external.topazlabs_upscale import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gemini-2", queue="cpu")
def gemini_2(args, **kwargs):
    from images.external.gemini_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gemini-3", queue="cpu")
def gemini_3(args, **kwargs):
    from images.external.gemini_3 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="seedream-4", queue="cpu")
def seedream_4(args, **kwargs):
    from images.external.seedream_4 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)
