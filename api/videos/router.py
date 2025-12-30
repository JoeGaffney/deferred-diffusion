from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from common.auth import verify_token
from common.schemas import DeleteResponse, Identity, TaskStatus
from common.storage import promote_result_to_storage
from utils.utils import cancel_task
from videos.schemas import (
    MODEL_META,
    VideoCreateResponse,
    VideoModelsResponse,
    VideoRequest,
    VideoResponse,
    VideoWorkerResponse,
    generate_model_docs,
)
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"], dependencies=[Depends(verify_token)])


@router.post("", response_model=VideoCreateResponse, operation_id="videos_create", description=generate_model_docs())
def create(video_request: VideoRequest, response: Response, identity: Identity = Depends(verify_token)):
    try:
        result = celery_app.send_task(
            video_request.task_name,
            queue=video_request.task_queue,
            args=[video_request.model_dump()],
            kwargs=identity.model_dump(),
        )
        response.headers["Location"] = f"/videos/{result.id}"
        return VideoCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get(
    "/models", response_model=VideoModelsResponse, summary="List video models", operation_id="videos_list_models"
)
def models():
    return VideoModelsResponse(models={name: meta for name, meta in MODEL_META.items()})


@router.get("/{id}", response_model=VideoResponse, operation_id="videos_get")
def get(id: UUID, identity: Identity = Depends(verify_token)):
    result = AsyncResult(str(id), app=celery_app)

    # Initialize response with common fields
    response = VideoResponse(
        id=id,
        status=result.status,
    )

    if result.info:
        if isinstance(result.info, dict):
            response.logs = result.info.get("logs", [])

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = result.result
            if not isinstance(result_data, dict):
                raise ValueError("Result data is not a dictionary")

            # Lazy Cache to Disk and get Signed URL
            download_url = promote_result_to_storage(
                id, result_data.get("base64_data"), "mp4", base_url=identity.base_url
            )

            # Return the URL and the base64_data
            response.result = VideoWorkerResponse(
                url=download_url,
                base64_data=result_data.get("base64_data"),
                logs=result_data.get("logs", []),
            )
            response.logs = response.result.logs
        except Exception as e:
            response.status = TaskStatus.FAILURE
            response.error_message = f"Error promoting result to storage: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="videos_delete")
def delete(id: UUID):
    return cancel_task(id, celery_app)
