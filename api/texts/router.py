from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from common.auth import verify_token
from common.schemas import DeleteResponse
from texts.schemas import (
    MODEL_META,
    TextCreateResponse,
    TextModelsResponse,
    TextRequest,
    TextResponse,
    TextWorkerResponse,
    generate_model_docs,
)
from utils.utils import cancel_task
from worker import celery_app

router = APIRouter(prefix="/texts", tags=["Texts"], dependencies=[Depends(verify_token)])


@router.post("", response_model=TextCreateResponse, operation_id="texts_create", description=generate_model_docs())
def create(request: TextRequest, response: Response):
    try:
        result = celery_app.send_task(request.task_name, queue=request.task_queue, args=[request.model_dump()])
        response.headers["Location"] = f"/texts/{result.id}"
        return TextCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/models", response_model=TextModelsResponse, summary="List text models", operation_id="texts_list_models")
def models():
    return TextModelsResponse(models={name: meta for name, meta in MODEL_META.items()})


@router.get("/{id}", response_model=TextResponse, operation_id="texts_get")
def get(id: UUID):
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


@router.delete("/{id}", response_model=DeleteResponse, operation_id="texts_delete")
def delete(id: UUID):
    return cancel_task(id, celery_app)
