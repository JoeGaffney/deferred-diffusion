import importlib
from typing import Any, Dict, Tuple

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


def model_router_main(context: ImageContext) -> Image.Image:
    """Route to the specific model implementation by concrete model name.

    Lazy-imports the module/attribute that the corresponding celery task would call.
    """
    model = context.data.model

    MODEL_NAME_TO_CALLABLE: Dict[ModelName, Tuple[str, str]] = {
        "sd-xl": ("images.models.sdxl", "main"),
        "sd-3": ("images.models.sd3", "main"),
        "flux-1": ("images.models.flux", "main"),
        "qwen-image": ("images.models.qwen", "main"),
        "depth-anything-2": ("images.models.depth_anything", "main"),
        "segment-anything-2": ("images.models.segment_anything", "main"),
        "real-esrgan-x4": ("images.models.real_esrgan", "main"),
        # external implementations (match celery task targets)
        "gpt-image-1": ("images.external_models.openai", "main"),
        "runway-gen4-image": ("images.external_models.runway", "main"),
        "flux-1-pro": ("images.external_models.flux", "main"),
        "topazlabs-upscale": ("images.external_models.topazlabs", "main"),
        "google-gemini-2": ("images.external_models.google_gemini", "main"),
        "bytedance-seedream-4": ("images.external_models.bytedance", "main"),
    }

    if model not in MODEL_NAME_TO_CALLABLE:
        raise ValueError(f"No direct model implementation mapped for model '{model}'")

    module_path, attr = MODEL_NAME_TO_CALLABLE[model]
    mod = importlib.import_module(module_path)
    main_fn = getattr(mod, attr)
    return main_fn(context)


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=name, queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="sd-3", queue="gpu")
def sd_3(request_dict):
    from images.models.sd3 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1", queue="gpu")
def flux_1(request_dict):
    from images.models.flux import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="qwen-image", queue="gpu")
def qwen_image(request_dict):
    from images.models.qwen import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="depth-anything-2", queue="gpu")
def depth_anything_2(request_dict):
    from images.models.depth_anything import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="segment-anything-2", queue="gpu")
def segment_anything_2(request_dict):
    from images.models.segment_anything import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="real-esrgan-x4", queue="gpu")
def real_esrgan_x4(request_dict):
    from images.models.real_esrgan import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sd-xl", queue="gpu")
def sd_xl(request_dict):
    from images.models.sdxl import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="gpt-image-1", queue="cpu")
def gpt_image_1(request_dict):
    from images.external_models.openai import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-gen4-image", queue="cpu")
def runway_gen4_image(request_dict):
    from images.external_models.runway import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="flux-1-pro", queue="cpu")
def flux_1_pro(request_dict):
    from images.external_models.flux import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="topazlabs-upscale", queue="cpu")
def topazlabs_upscale(request_dict):
    from images.external_models.topazlabs import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="google-gemini-2", queue="cpu")
def google_gemini_2(request_dict):
    from images.external_models.google_gemini import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="bytedance-seedream-4", queue="cpu")
def bytedance_seedream_4(request_dict):
    from images.external_models.bytedance import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
