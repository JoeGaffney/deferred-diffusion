import asyncio

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from utils.utils import mp4_to_base64
from videos.context import VideoContext
from videos.models.hunyuan_video import main as hunyuan_video_main
from videos.models.ltx_video import main as ltx_video_main
from videos.models.runway_gen import main as runway_gen_main
from videos.models.wan_2_1 import main as wan_2_1_main
from videos.schemas import VideoRequest, VideoResponse

router = APIRouter(prefix="/videos", tags=["Videos"])

from worker import celery_app  # Import from worker.py


@router.post("", response_model=VideoResponse, operation_id="videos_create")
async def create(request: VideoRequest):
    context = VideoContext(request)

    main = None
    if request.model == "LTX-Video":
        main = ltx_video_main
    elif request.model == "HunyuanVideo":
        main = hunyuan_video_main
    elif request.model == "Wan2.1":
        main = wan_2_1_main
    elif request.model == "runway/gen3a_turbo" or request.model == "runway/gen4_turbo":
        main = runway_gen_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return VideoResponse(base64_data=mp4_to_base64(result))


@router.post("/celery_test", operation_id="videos_celery_test")
async def celery_test():
    celery_task = celery_app.send_task("create_task", args=[int(10)])
    # celery_task_2 = celery_app.send_task("process_video", args=[request])
    print(celery_task)
    # Return only the serializable parts of the task
    return {"task_id": celery_task.id, "status": celery_task.status, "state": celery_task.state}


@router.get("/tasks/{task_id}")
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {"task_id": task_id, "task_status": task_result.status, "task_result": task_result.result}
    return JSONResponse(result)
