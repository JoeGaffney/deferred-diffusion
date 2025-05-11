from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, Response

from images.schemas import (
    ImageCreateResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
)
from utils.utils import poll_task_until_complete
from worker import celery_app

router = APIRouter(prefix="/images", tags=["Images"])


@router.post("", response_model=ImageCreateResponse, operation_id="images_create")
async def create(request: ImageRequest, response: Response):
    try:
        task_result = celery_app.send_task("process_image", args=[request.model_dump()])
        response.headers["Location"] = f"/images/{task_result.id}"
        return ImageCreateResponse(task_id=task_result.id, task_status=task_result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/{task_id}", response_model=ImageResponse, operation_id="images_get")
async def get(task_id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        task_result = await poll_task_until_complete(task_id)
    else:
        task_result = AsyncResult(task_id)

    # Initialize response with common fields
    response = ImageResponse(
        task_id=task_id,
        task_status=task_result.status,
    )

    # Add appropriate fields based on status
    if task_result.successful():
        try:
            result = ImageWorkerResponse.model_validate(task_result.result)
            response.task_result = result
        except Exception as e:
            response.task_status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif task_result.failed():
        response.error_message = f"Task failed with error: {str(task_result.result)}"

    return response
