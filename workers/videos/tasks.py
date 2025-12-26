from common.logger import get_task_logs
from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest, VideoWorkerResponse
from worker import celery_app


def process_result(context, result):
    """Process the result of video generation."""
    if result:
        return VideoWorkerResponse(base64_data=mp4_to_base64(result), logs=get_task_logs()).model_dump()
    raise ValueError("Video generation failed")


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(args, **kwargs):
    request = VideoRequest.model_validate(args)
    context = VideoContext(request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=f"videos.{name}", queue=queue)


# Explicit internal model tasks (lazy-import model implementation inside each task)
@typed_task(name="ltx-video", queue="gpu")
def ltx_video(args, **kwargs):
    from videos.local.ltx_video import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="wan-2", queue="gpu")
def wan_2(args, **kwargs):
    from videos.local.wan_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="hunyuan-video-1", queue="gpu")
def hunyuan_video_1(args, **kwargs):
    from videos.local.hunyuan_video_1 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sam-3", queue="gpu")
def sam_3(args, **kwargs):
    from videos.local.sam_3 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


# Explicit external model tasks
@typed_task(name="runway-gen-4", queue="cpu")
def runway_gen_4(args, **kwargs):
    from videos.external.runway_gen_4 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="runway-upscale", queue="cpu")
def runway_upscale(args, **kwargs):
    from videos.external.runway_upscale import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="seedance-1", queue="cpu")
def seedance_1(args, **kwargs):
    from videos.external.seedance_1 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="kling-2", queue="cpu")
def kling_2(args, **kwargs):
    from videos.external.kling_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="veo-3", queue="cpu")
def veo_3(args, **kwargs):
    from videos.external.veo_3 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="sora-2", queue="cpu")
def sora_2(args, **kwargs):
    from videos.external.sora_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)


@typed_task(name="hailuo-2", queue="cpu")
def hailuo_2(args, **kwargs):
    from videos.external.hailuo_2 import main

    context = validate_request_and_context(args)
    result = main(context)
    return process_result(context, result)
