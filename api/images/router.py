import os
from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from common.auth import verify_token
from common.comfy.comfy_utils import model_schema_to_comfy_nodes
from common.logger import log_pretty
from common.schemas import ComfyWorkflowResponse
from images.schemas import (
    ImageCreateResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
)
from utils.utils import poll_until_complete
from worker import celery_app

router = APIRouter(
    prefix="/images", tags=["Images"], dependencies=[Depends(verify_token)]  # This will apply to all routes
)


@router.post("", response_model=ImageCreateResponse, operation_id="images_create")
async def create(request: ImageRequest, response: Response):
    task_name = "process_image"
    if request.comfy_workflow:
        task_name = "process_image_workflow"
    if request.external_model:
        task_name = "process_image_external"

    try:
        result = celery_app.send_task(task_name, args=[request.model_dump()])
        response.headers["Location"] = f"/images/{result.id}"
        return ImageCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/workflow/schema", response_model=ComfyWorkflowResponse, operation_id="images_get_workflow_schema")
async def get_workflow_schema():
    result = model_schema_to_comfy_nodes(ImageRequest)

    log_pretty(f"Generated for ImageRequest schema", result.model_dump())
    return result


@router.get("/{id}", response_model=ImageResponse, operation_id="images_get")
async def get(id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        result = await poll_until_complete(str(id))
    else:
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
