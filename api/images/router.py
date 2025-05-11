from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query, Response

from images.schemas import (
    ImageCreateResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
)
from utils.utils import poll_until_complete
from worker import celery_app

router = APIRouter(prefix="/images", tags=["Images"])


@router.post("", response_model=ImageCreateResponse, operation_id="images_create")
async def create(request: ImageRequest, response: Response):
    try:
        result = celery_app.send_task("process_image", args=[request.model_dump()])
        response.headers["Location"] = f"/images/{result.id}"
        return ImageCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/{id}", response_model=ImageResponse, operation_id="images_get")
async def get(id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        result = await poll_until_complete(id)
    else:
        result = AsyncResult(id)

    # Initialize response with common fields
    response = ImageResponse(
        id=id,
        status=result.status,
    )

    # Add appropriate fields based on status
    if result.successful():
        try:
            result = ImageWorkerResponse.model_validate(result.result)
            response.result = result
        except Exception as e:
            response.status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response
