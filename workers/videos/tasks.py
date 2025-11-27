from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest, VideoWorkerResponse
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


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=f"videos.{name}", queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="ltx-video", queue="gpu")
def ltx_video(request_dict):
    from videos.local.ltx_video import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="wan-2", queue="gpu")
def wan_2(request_dict):
    from videos.local.wan_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="runway-gen-4", queue="cpu")
def runway_gen_4(request_dict):
    from videos.external.runway_gen_4 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-act-two", queue="cpu")
def runway_act_two(request_dict):
    from videos.external.runway_act_two import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-upscale", queue="cpu")
def runway_upscale(request_dict):
    from videos.external.runway_upscale import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="bytedance-seedance-1", queue="cpu")
def bytedance_seedance_1(request_dict):
    from videos.external.bytedance_seedance_1 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="kwaivgi-kling-2", queue="cpu")
def kwaivgi_kling_2(request_dict):
    from videos.external.kwaivgi_kling_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="google-veo-3", queue="cpu")
def google_veo_3(request_dict):
    from videos.external.google_veo_3 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="openai-sora-2", queue="cpu")
def openai_sora_2(request_dict):
    from videos.external.openai_sora_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="minimax-hailuo-2", queue="cpu")
def minimax_hailuo_2(request_dict):
    from videos.external.minimax_hailuo_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
