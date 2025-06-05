from common.memory import free_gpu_memory
from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.models.ltx_video import main as ltx_video_main
from videos.models.runway_gen import main as runway_gen_main
from videos.models.wan_2_1 import main as wan_2_1_main
from videos.schemas import VideoRequest, VideoWorkerResponse
from worker import celery_app


@celery_app.task(name="process_video")
def process_video(request_dict):
    free_gpu_memory()
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(request)

    result = None
    if context.model == "LTX-Video":
        result = ltx_video_main(context)
    elif context.model == "Wan2.1":
        result = wan_2_1_main(context)
    else:
        raise ValueError(f"Unsupported model: {context.model}")

    return VideoWorkerResponse(base64_data=mp4_to_base64(result)).model_dump()


@celery_app.task(name="process_video_external")
def process_video_external(request_dict):
    request = VideoRequest.model_validate(request_dict)
    context = VideoContext(request)

    result = None
    if context.model == "runway/gen3a_turbo" or context.model == "runway/gen4_turbo":
        result = runway_gen_main(context)
    else:
        raise ValueError(f"Unsupported model: {context.model}")

    return VideoWorkerResponse(base64_data=mp4_to_base64(result)).model_dump()
