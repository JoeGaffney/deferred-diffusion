from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from common.auth import verify_token
from common.comfy.comfy_utils import model_schema_to_comfy_nodes
from common.logger import log_pretty
from common.schemas import ComfyWorkflowResponse
from utils.utils import poll_until_resolved
from videos.schemas import (
    VideoCreateResponse,
    VideoRequest,
    VideoResponse,
    VideoWorkerResponse,
)
from worker import celery_app

router = APIRouter(prefix="/videos", tags=["Videos"], dependencies=[Depends(verify_token)])


@router.post("", response_model=VideoCreateResponse, operation_id="videos_create")
async def create(request: VideoRequest, response: Response):
    try:
        result = celery_app.send_task(request.task_name, args=[request.model_dump()])
        response.headers["Location"] = f"/videos/{result.id}"
        return VideoCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/workflow/schema", response_model=ComfyWorkflowResponse, operation_id="videos_get_workflow_schema")
async def get_workflow_schema():
    result = model_schema_to_comfy_nodes(VideoRequest)

    log_pretty(f"Generated for VideoRequest schema", result.model_dump())
    return result


@router.get("/{id}", response_model=VideoResponse, operation_id="videos_get")
async def get(id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        result = await poll_until_resolved(str(id), timeout=1000, poll_interval=10)
    else:
        result = AsyncResult(id, app=celery_app)

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
