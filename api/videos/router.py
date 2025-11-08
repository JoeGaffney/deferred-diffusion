from typing import Dict, Literal, Optional
from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from common.auth import verify_token
from videos.schemas import (
    MODEL_META,
    InferredMode,
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
def create(request: VideoRequest, response: Response):
    try:
        result = celery_app.send_task(request.task_name, queue=request.task_queue, args=[request.model_dump()])
        response.headers["Location"] = f"/videos/{result.id}"
        return VideoCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get(
    "/models", response_model=VideoModelsResponse, summary="List video models", operation_id="videos_list_models"
)
def models(
    mode: Optional[InferredMode] = Query(default=None),
    external: Optional[bool] = Query(default=None),
):
    return VideoModelsResponse(
        models={
            name: meta
            for name, meta in MODEL_META.items()
            if (external is None or meta.external == external) and (mode is None or meta.supports_inferred_mode(mode))
        }
    )


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
