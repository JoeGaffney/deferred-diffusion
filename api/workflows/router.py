from uuid import UUID

from fastapi import APIRouter, Depends

from common.auth import verify_token
from common.schemas import DeleteResponse, Identity
from common.storage import signed_url_for_file
from common.task_helpers import cancel_task, create_task, get_task_detailed
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
def create(workflow_request: WorkflowRequest, identity: Identity = Depends(verify_token)):
    result = create_task(
        workflow_request.task_name,
        "comfy",
        workflow_request.model_dump(),
        identity,
    )
    return WorkflowCreateResponse(id=UUID(str(result.id)), status=result.status)


@router.get("/{id}", response_model=WorkflowResponse, operation_id="workflows_get")
def get(id: UUID):
    result, task_info, logs = get_task_detailed(id)

    # Initialize response with common fields
    response = WorkflowResponse(id=id, status=result.status, task_info=task_info, logs=logs)

    # Add appropriate fields based on status
    if result.successful():
        result_data = WorkflowWorkerResponse.model_validate(result.result)
        response.logs = result_data.logs

        # Convert all file paths to signed URLs
        response.output = [signed_url_for_file(file_id) for file_id in result_data.output]
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="workflows_delete")
def delete(id: UUID):
    return cancel_task(id)
