from PIL import Image

from common.logger import get_task_logs
from utils.utils import pil_to_base64
from worker import celery_app
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowRequest, WorkflowWorkerResponse


def process_result(context, result):
    if isinstance(result, Image.Image):
        context.save_image(result)
        return WorkflowWorkerResponse(base64_data=pil_to_base64(result), logs=get_task_logs()).model_dump()
    raise ValueError("Image generation failed")


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = WorkflowRequest.model_validate(request_dict)
    context = WorkflowContext(request)
    return context


@celery_app.task(name="comfy-workflow", queue="comfy")
def comfy_workflow(request_dict):
    from workflows.comfy.comfy_workflow import main

    context = validate_request_and_context(request_dict)
    result = main(context)

    return process_result(context, result)
