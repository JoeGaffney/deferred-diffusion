from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Response

from common.auth import verify_token
from common.logger import logger
from common.schemas import DeleteResponse, TaskStatus
from images.schemas import (
    MODEL_META,
    ImageCreateResponse,
    ImageModelsResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
    generate_model_docs,
)
from utils.utils import cancel_task
from worker import celery_app

router = APIRouter(
    prefix="/images", tags=["Images"], dependencies=[Depends(verify_token)]  # This will apply to all routes
)


@router.post(
    "",
    response_model=ImageCreateResponse,
    description=generate_model_docs(),
    operation_id="images_create",
)
def create(request: ImageRequest, response: Response):
    try:
        result = celery_app.send_task(request.task_name, queue=request.task_queue, args=[request.model_dump()])
        response.headers["Location"] = f"/images/{result.id}"
        return ImageCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get(
    "/models", response_model=ImageModelsResponse, summary="List image models", operation_id="images_list_models"
)
def models():
    return ImageModelsResponse(models={name: meta for name, meta in MODEL_META.items()})


@router.get("/{id}", response_model=ImageResponse, operation_id="images_get")
def get(id: UUID):
    result = AsyncResult(str(id), app=celery_app)

    # Initialize response with common fields
    response = ImageResponse(
        id=id,
        status=result.status,
    )

    if result.info:
        if isinstance(result.info, dict):
            response.logs = result.info.get("logs", [])

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = ImageWorkerResponse.model_validate(result.result)
            response.result = result_data
            response.logs = result_data.logs
        except Exception as e:
            response.status = TaskStatus.FAILURE
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="images_delete")
def delete(id: UUID):
    return cancel_task(id, celery_app)
