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
def validate_request_and_context(model: ModelName, request_dict):
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(model, request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=name, queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="ltx-video", queue="gpu")
def ltx_video(request_dict):
    from videos.models.ltx import main

    context = validate_request_and_context("ltx-video", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="wan-2", queue="gpu")
def wan_2(request_dict):
    from videos.models.wan import main

    context = validate_request_and_context("wan-2", request_dict)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="runway-gen-4", queue="cpu")
def runway_gen_4(request_dict):
    from videos.external_models.runway import main

    context = validate_request_and_context("runway-gen-4", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-act-two", queue="cpu")
def runway_act_two(request_dict):
    from videos.external_models.runway_act import main

    context = validate_request_and_context("runway-act-two", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-upscale", queue="cpu")
def runway_upscale(request_dict):
    from videos.external_models.runway_upscale import main

    context = validate_request_and_context("runway-upscale", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-gen-4-aleph", queue="cpu")
def runway_gen_4_aleph(request_dict):
    from videos.external_models.runway_aleph import main

    context = validate_request_and_context("runway-gen-4-aleph", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="bytedance-seedance-1", queue="cpu")
def bytedance_seedance_1(request_dict):
    from videos.external_models.bytedance_seedance import main

    context = validate_request_and_context("bytedance-seedance-1", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="kwaivgi-kling-2", queue="cpu")
def kwaivgi_kling_2(request_dict):
    from videos.external_models.kling import main

    context = validate_request_and_context("kwaivgi-kling-2", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="google-veo-3", queue="cpu")
def google_veo_3(request_dict):
    from videos.external_models.google_veo import main

    context = validate_request_and_context("google-veo-3", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="openai-sora-2", queue="cpu")
def openai_sora_2(request_dict):
    from videos.external_models.openai import main

    context = validate_request_and_context("openai-sora-2", request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="minimax-hailuo-2", queue="cpu")
def minimax_hailuo_2(request_dict):
    from videos.external_models.minimax_hailuo import main

    context = validate_request_and_context("minimax-hailuo-2", request_dict)
    result = main(context)
    return process_result(context, result)
