import time

from celery import Celery, shared_task

from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.models.hunyuan_video import main as hunyuan_video_main
from videos.models.ltx_video import main as ltx_video_main
from videos.models.runway_gen import main as runway_gen_main
from videos.models.wan_2_1 import main as wan_2_1_main
from videos.schemas import VideoRequest
from worker import celery_app  # Import from worker.py


@celery_app.task(name="process_video")
def process_video(request: VideoRequest):
    """Process video generation asynchronously"""
    # Recreate context from dict
    context = VideoContext(request)

    main = None
    if context.model == "LTX-Video":
        main = ltx_video_main
    elif context.model == "HunyuanVideo":
        main = hunyuan_video_main
    elif context.model == "Wan2.1":
        main = wan_2_1_main
    elif context.model == "runway/gen3a_turbo" or context.model == "runway/gen4_turbo":
        main = runway_gen_main

    if not main:
        raise ValueError(f"Invalid model {context.model}")

    # Process video
    result = main(context)
    # Return base64 encoded result
    return mp4_to_base64(result)


@celery_app.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True
