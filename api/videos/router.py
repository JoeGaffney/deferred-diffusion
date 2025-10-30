from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from common.auth import verify_token
from videos.schemas import (
    ModelNameExternal,
    ModelNameLocal,
    VideoCreateResponse,
    VideoRequest,
    VideoResponse,
    VideoWorkerResponse,
    generate_model_docs,
)
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"], dependencies=[Depends(verify_token)])


def _create_task(model: str, queue: str, request: VideoRequest, response: Response) -> VideoCreateResponse:
    try:
        result = celery_app.send_task(model, queue=queue, args=[request.model_dump()])
        response.headers["Location"] = f"/videos/{result.id}"
        return VideoCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.post(
    "/local/{model}",
    response_model=VideoCreateResponse,
    operation_id="videos_create_local",
    description=generate_model_docs(local=True),
)
def create_local(model: ModelNameLocal, request: VideoRequest, response: Response):
    return _create_task(model, "gpu", request, response)


@router.post(
    "/external/{model}",
    response_model=VideoCreateResponse,
    operation_id="videos_create_external",
    description=generate_model_docs(local=False),
)
def create_external(model: ModelNameExternal, request: VideoRequest, response: Response):
    return _create_task(model, "cpu", request, response)


@router.get("/{id}", response_model=VideoResponse, operation_id="videos_get")
def get(id: UUID):
    result = AsyncResult(str(id), app=celery_app)

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
