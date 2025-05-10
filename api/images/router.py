from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from images.schemas import GetImageResponse, ImageRequest, ImageResponse
from utils.utils import poll_task_until_complete
from worker import celery_app

router = APIRouter(prefix="/images", tags=["Images"])


@router.post("", response_model=ImageResponse, operation_id="images_create")
async def create(request: ImageRequest):
    task_id = celery_app.send_task("process_image", args=[request.model_dump()])

    # Wait for the result
    task_result = await poll_task_until_complete(task_id)

    # Convert the dictionary result back to a Pydantic model
    try:
        return ImageResponse.model_validate(task_result.result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")


@router.get("/{task_id}", response_model=GetImageResponse, operation_id="images_get")
async def get(task_id, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        # Wait for the task to complete
        task_result = await poll_task_until_complete(task_id)
    else:
        # Just get the current status without waiting
        task_result = AsyncResult(task_id)

    # Initialize response with common fields
    response = GetImageResponse(
        task_id=task_id,
        task_status=task_result.status,
    )

    # Add appropriate fields based on status
    if task_result.successful():
        try:
            # Parse the result into your ImageResponse model
            image_response = ImageResponse.model_validate(task_result.result)
            response.task_result = image_response
        except Exception as e:
            # Handle validation errors
            response.task_status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif task_result.failed():
        # Include error details
        response.error_message = str(task_result.result)

    return response
