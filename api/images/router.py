from typing import Dict, Literal, Optional
from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from common.auth import verify_token
from images.schemas import (
    MODEL_META,
    ImageCreateResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
    InferredMode,
    ModelInfo,
    ModelName,
    generate_model_docs,
)
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
    "/models",
    response_model=Dict[ModelName, ModelInfo],
    summary="List image models",
    operation_id="images_list_models",
)
def list_image_models(
    mode: Optional[InferredMode] = Query(default=None),
    external: Optional[bool] = Query(default=None),
):
    # Filter by capability/provider/externality
    return {
        name: meta
        for name, meta in MODEL_META.items()
        if (external is None or meta.external == external) and (mode is None or meta.supports_inferred_mode(mode))
    }


@router.get(
    "/models/docs",
    response_class=PlainTextResponse,
    summary="Image model capability docs (Markdown)",
    operation_id="images_model_docs",
)
def image_model_docs():
    return generate_model_docs()


@router.get("/{id}", response_model=ImageResponse, operation_id="images_get")
def get(id: UUID):
    result = AsyncResult(str(id), app=celery_app)

    # Initialize response with common fields
    response = ImageResponse(
        id=id,
        status=result.status,
    )

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = ImageWorkerResponse.model_validate(result.result)
            response.result = result_data
        except Exception as e:
            response.status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response
