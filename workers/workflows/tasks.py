from PIL import Image

from common.logger import get_task_logs
from common.memory import free_gpu_memory
from common.pipeline_helpers import clear_global_pipeline_cache
from utils.utils import pil_to_base64
from worker import celery_app
from workflows.comfy.comfy import main as comfy_main
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


@celery_app.task(name="workflow")
def workflow(request_dict):
    clear_global_pipeline_cache()
    free_gpu_memory()

    context = validate_request_and_context(request_dict)

    result = comfy_main(context)

    # free up GPU memory again after processing
    free_gpu_memory()
    return process_result(context, result)
