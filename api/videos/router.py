from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from common.auth import verify_token
from utils.utils import poll_until_complete
from videos.schemas import (
    VideoCreateResponse,
    VideoRequest,
    VideoResponse,
    VideoWorkerResponse,
)
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"], dependencies=[Depends(verify_token)])


@router.post("", response_model=VideoCreateResponse, operation_id="videos_create")
async def create(request: VideoRequest, response: Response):
    task_name = "process_video"
    if request.external_model:
        task_name = "process_video_external"

    try:
        result = celery_app.send_task(task_name, args=[request.model_dump()])
        response.headers["Location"] = f"/videos/{result.id}"
        return VideoCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/{id}", response_model=VideoResponse, operation_id="videos_get")
async def get(id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        result = await poll_until_complete(str(id))
    else:
        result = AsyncResult(id, app=celery_app)

    # Initialize response with common fields
    response = VideoResponse(
        id=id,
        status=result.status,
    )

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = VideoWorkerResponse.model_validate(result.result)
            response.result = result_data
        except Exception as e:
            response.status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response
