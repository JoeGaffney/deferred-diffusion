from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from common.auth import verify_token
from common.schemas import DeleteResponse, Identity
from common.task_helpers import cancel_task, get_task_detailed
from texts.schemas import (
    MODEL_META,
    TextCreateResponse,
    TextModelsResponse,
    TextRequest,
    TextResponse,
    TextWorkerResponse,
    generate_model_docs,
)
from worker import celery_app

router = APIRouter(prefix="/texts", tags=["Texts"], dependencies=[Depends(verify_token)])


@router.post("", response_model=TextCreateResponse, operation_id="texts_create", description=generate_model_docs())
def create(text_request: TextRequest, response: Response, identity: Identity = Depends(verify_token)):
    try:
        result = celery_app.send_task(
            text_request.task_name,
            queue=text_request.task_queue,
            args=[text_request.model_dump()],
            kwargs=identity.model_dump(),
        )
        response.headers["Location"] = f"/texts/{result.id}"
        return TextCreateResponse(id=result.id, status=result.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.get("/models", response_model=TextModelsResponse, summary="List text models", operation_id="texts_list_models")
def models():
    return TextModelsResponse(models={name: meta for name, meta in MODEL_META.items()})


@router.get("/{id}", response_model=TextResponse, operation_id="texts_get")
def get(id: UUID):
    result, task_info, logs = get_task_detailed(id, celery_app)

    # Initialize response with common fields
    response = TextResponse(
        id=id,
        status=result.status,
        task_info=task_info,
        logs=logs,
    )

    # Add appropriate fields based on status
    if result.successful():
        result_data = TextWorkerResponse.model_validate(result.result)
        response.output = result_data.response
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="texts_delete")
def delete(id: UUID):
    return cancel_task(id, celery_app)
