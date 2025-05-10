from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from utils.utils import poll_task_until_complete
from videos.schemas import VideoRequest, VideoResponse
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("", response_model=VideoResponse, operation_id="videos_create")
async def create(request: VideoRequest):
    task_id = celery_app.send_task("process_video", args=request.model_dump())
    task_result = await poll_task_until_complete(task_id)

    try:
        return VideoResponse(base64_data=task_result.result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")


@router.get("/{task_id}", operation_id="videos_get")
async def get(task_id, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        # Wait for the task to complete
        task_result = await poll_task_until_complete(task_id)
    else:
        # Just get the current status without waiting
        task_result = AsyncResult(task_id)

    result = {"task_id": task_id, "task_status": task_result.status, "task_result": task_result.result}
    return JSONResponse(result)
