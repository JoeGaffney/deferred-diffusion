from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from images.schemas import ImageRequest, ImageResponse
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


@router.get("/{task_id}", operation_id="images_get")
async def get(task_id, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        # Wait for the task to complete
        task_result = await poll_task_until_complete(task_id)
    else:
        # Just get the current status without waiting
        task_result = AsyncResult(task_id)

    result = {"task_id": task_id, "task_status": task_result.status, "task_result": task_result.result}
    return JSONResponse(result)
