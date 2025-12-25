from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Response

from common.auth import verify_token
from common.schemas import DeleteResponse, TaskStatus
from utils.utils import cancel_task
from worker import celery_app
from workflows.schemas import (
    WorkflowCreateResponse,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowWorkerResponse,
)

router = APIRouter(
    prefix="/workflows", tags=["Workflows"], dependencies=[Depends(verify_token)]  # This will apply to all routes
)


@router.post(
    "",
    response_model=WorkflowCreateResponse,
    description="ComfyUI workflow execution endpoint. Requires workflow JSON Api workflow and patches to swap out user data.",
    operation_id="workflows_create",
)
def create(request: WorkflowRequest, response: Response):
    try:
        result = celery_app.send_task(request.task_name, queue="comfy", args=[request.model_dump()])
        response.headers["Location"] = f"/workflows/{result.id}"
        return WorkflowCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/{id}", response_model=WorkflowResponse, operation_id="workflows_get")
def get(id: UUID):
    result = AsyncResult(str(id), app=celery_app)

    # Initialize response with common fields
    response = WorkflowResponse(
        id=id,
        status=result.status,
    )

    if result.info:
        if isinstance(result.info, dict):
            response.logs = result.info.get("logs", [])

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = WorkflowWorkerResponse.model_validate(result.result)
            response.result = result_data
            response.logs = result_data.logs
        except Exception as e:
            response.status = TaskStatus.FAILURE
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="workflows_delete")
def delete(id: UUID):
    return cancel_task(id, celery_app)
