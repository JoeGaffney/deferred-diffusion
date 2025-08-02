from common.memory import free_gpu_memory
from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.external_models.runway import main as runway_main
from videos.external_models.runway_act import main as runway_act_main
from videos.external_models.runway_aleph import main as runway_aleph_main
from videos.external_models.runway_upscale import main as runway_upscale_main
from videos.models.ltx import main as ltx_main
from videos.models.wan import main as wan_main
from videos.schemas import VideoRequest, VideoWorkerResponse
from worker import celery_app


def process_result(context, result):
    """Process the result of video generation."""
    if result:
        return VideoWorkerResponse(base64_data=mp4_to_base64(result)).model_dump()
    raise ValueError("Video generation failed")


def model_router_main(context: VideoContext):
    family = context.data.model_family
    if family == "ltx":
        return ltx_main(context)
    elif family == "wan":
        return wan_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")


def external_model_router_main(context: VideoContext):
    family = context.data.model_family
    if family == "runway":
        return runway_main(context)
    elif family == "runway_act":
        return runway_act_main(context)
    elif family == "runway_upscale":
        return runway_upscale_main(context)
    elif family == "runway_aleph":
        return runway_aleph_main(context)
    else:
        raise ValueError(f"Unsupported model family: {family}")


@celery_app.task(name="process_video", queue="gpu")
def process_video(request_dict):
    free_gpu_memory()
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(request)
    result = model_router_main(context)

    return process_result(context, result)


@celery_app.task(name="process_video_external", queue="cpu")
def process_video_external(request_dict):
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(request)
    result = external_model_router_main(context)

    return process_result(context, result)
