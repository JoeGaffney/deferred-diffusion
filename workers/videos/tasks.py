import importlib

from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.schemas import VideoRequest, VideoWorkerResponse
from worker import celery_app


def process_result(context, result):
    """Process the result of video generation."""
    if result:
        return VideoWorkerResponse(base64_data=mp4_to_base64(result)).model_dump()
    raise ValueError("Video generation failed")


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(request)
    return context


def model_router_main(context: VideoContext):
    """Route to the specific model implementation by concrete model name.

    Lazy-imports the module/attribute that the corresponding celery task would call.
    """
    model = context.data.model

    MODEL_NAME_TO_CALLABLE = {
        "ltx-video": ("videos.models.ltx", "main"),
        "wan-2-1": ("videos.models.wan", "main"),
        "wan-2-2": ("videos.models.wan", "main"),
        # external implementations (match celery task targets)
        "runway-gen-3": ("videos.external_models.runway", "main"),
        "runway-gen-4": ("videos.external_models.runway", "main"),
        "runway-act-two": ("videos.external_models.runway_act", "main"),
        "runway-upscale": ("videos.external_models.runway_upscale", "main"),
        "runway-gen-4-aleph": ("videos.external_models.runway_aleph", "main"),
        "bytedance-seedance-1": ("videos.external_models.bytedance_seedance", "main"),
        "kwaivgi-kling-2-1": ("videos.external_models.kling", "main"),
        "google-veo-3": ("videos.external_models.google_veo", "main"),
        "minimax-hailuo-2": ("videos.external_models.minimax_hailuo", "main"),
    }

    if model not in MODEL_NAME_TO_CALLABLE:
        raise ValueError(f"No direct model implementation mapped for model '{model}'")

    module_path, attr = MODEL_NAME_TO_CALLABLE[model]
    mod = importlib.import_module(module_path)
    main_fn = getattr(mod, attr)
    return main_fn(context)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@celery_app.task(name="ltx-video", queue="gpu")
def ltx_video(request_dict):
    from videos.models.ltx import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="wan-2-1", queue="gpu")
def wan_2_1(request_dict):
    from videos.models.wan import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="wan-2-2", queue="gpu")
def wan_2_2(request_dict):
    from videos.models.wan import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@celery_app.task(name="runway-gen-3", queue="cpu")
def runway_gen_3(request_dict):
    from videos.external_models.runway import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="runway-gen-4", queue="cpu")
def runway_gen_4(request_dict):
    from videos.external_models.runway import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="runway-act-two", queue="cpu")
def runway_act_two(request_dict):
    from videos.external_models.runway_act import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="runway-upscale", queue="cpu")
def runway_upscale(request_dict):
    from videos.external_models.runway_upscale import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="runway-gen-4-aleph", queue="cpu")
def runway_gen_4_aleph(request_dict):
    from videos.external_models.runway_aleph import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="bytedance-seedance-1", queue="cpu")
def bytedance_seedance_1(request_dict):
    from videos.external_models.bytedance_seedance import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="kwaivgi-kling-2-1", queue="cpu")
def kwaivgi_kling_2_1(request_dict):
    from videos.external_models.kling import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="google-veo-3", queue="cpu")
def google_veo_3(request_dict):
    from videos.external_models.google_veo import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@celery_app.task(name="minimax-hailuo-2", queue="cpu")
def minimax_hailuo_2(request_dict):
    from videos.external_models.minimax_hailuo import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
