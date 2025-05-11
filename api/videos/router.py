from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from utils.utils import poll_until_complete
from videos.schemas import VideoRequest, VideoResponse
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("", response_model=VideoResponse, operation_id="videos_create")
async def create(request: VideoRequest):
    id = celery_app.send_task("process_video", args=request.model_dump())
    result = await poll_until_complete(id)

    try:
        return VideoResponse(base64_data=result.result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")


@router.get("/{id}", operation_id="videos_get")
async def get(id, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        # Wait for the task to complete
        result = await poll_until_complete(id)
    else:
        # Just get the current status without waiting
        result = AsyncResult(id)

    result = {"id": id, "status": result.status, "result": result.result}
    return JSONResponse(result)
