from uuid import UUID

from fastapi import APIRouter, Depends

from common.auth import verify_token
from common.schemas import DeleteResponse, Identity
from common.storage import signed_url_for_file
from common.task_helpers import cancel_task, create_task, get_task_detailed
from images.schemas import (
    MODEL_META,
    ImageCreateResponse,
    ImageModelsResponse,
    ImageRequest,
    ImageResponse,
    ImageWorkerResponse,
    generate_model_docs,
)

router = APIRouter(
    prefix="/images", tags=["Images"], dependencies=[Depends(verify_token)]  # This will apply to all routes
)


@router.post("", response_model=ImageCreateResponse, description=generate_model_docs(), operation_id="images_create")
def create(image_request: ImageRequest, identity: Identity = Depends(verify_token)):
    result = create_task(
        image_request.task_name,
        image_request.task_queue,
        image_request.model_dump(),
        identity,
    )
    return ImageCreateResponse(id=UUID(str(result.id)), status=result.status)


@router.get(
    "/models", response_model=ImageModelsResponse, summary="List image models", operation_id="images_list_models"
)
def models():
    return ImageModelsResponse(models={name: meta for name, meta in MODEL_META.items()})


@router.get("/{id}", response_model=ImageResponse, operation_id="images_get")
def get(id: UUID):
    result, task_info, logs = get_task_detailed(id)
    response = ImageResponse(id=id, status=result.status, task_info=task_info, logs=logs)

    # Add appropriate fields based on status
    if result.successful():
        result_data = ImageWorkerResponse.model_validate(result.result)
        response.logs = result_data.logs

        # Convert all file paths to signed URLs
        response.output = [signed_url_for_file(file_id) for file_id in result_data.output]
    elif result.failed():
        response.error_message = f"Task failed with error: {str(result.result)}"

    return response


@router.delete("/{id}", response_model=DeleteResponse, operation_id="images_delete")
def delete(id: UUID):
    return cancel_task(id)
