from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from common.auth import verify_token
from texts.schemas import (
    TextCreateResponse,
    TextRequest,
    TextResponse,
    TextWorkerResponse,
)
from utils.utils import poll_until_complete
from worker import celery_app

router = APIRouter(prefix="/texts", tags=["Texts"], dependencies=[Depends(verify_token)])


@router.post("", response_model=TextCreateResponse, operation_id="texts_create")
async def create(request: TextRequest, response: Response):
    task_name = "process_text"
    if request.external_model:
        task_name = "process_text_external"

    try:
        result = celery_app.send_task(task_name, args=[request.model_dump()])
        response.headers["Location"] = f"/texts/{result.id}"
        return TextCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/{id}", response_model=TextResponse, operation_id="texts_get")
async def get(id: UUID, wait: bool = Query(True, description="Whether to wait for task completion")):
    if wait:
        result = await poll_until_complete(str(id))
    else:
        result = AsyncResult(str(id), app=celery_app)

    # Initialize response with common fields
    response = TextResponse(
        id=id,
        status=result.status,
    )

    # Add appropriate fields based on status
    if result.successful():
        try:
            result_data = TextWorkerResponse.model_validate(result.result)
            response.result = result_data
        except Exception as e:
            response.status = "ERROR"
            response.error_message = f"Error parsing result: {str(e)}"
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response
