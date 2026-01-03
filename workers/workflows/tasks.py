from pathlib import Path
from typing import List

from common.config import settings
from common.logger import get_task_logs
from worker import celery_app
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowRequest, WorkflowWorkerResponse


def process_result(context: WorkflowContext, result: List[Path]):
    storage_dir = Path(settings.storage_dir)
    output = [str(path.relative_to(storage_dir)) for path in result]
    return WorkflowWorkerResponse(output=output, logs=get_task_logs()).model_dump()


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(args, **kwargs):
    request = WorkflowRequest.model_validate(args)
    context = WorkflowContext(request)
    return context


@celery_app.task(name="workflows.comfy-workflow", queue="comfy")
def comfy_workflow(args, **kwargs):
    from workflows.comfy.comfy_workflow import main

    context = validate_request_and_context(args)
    result = main(context)

    return process_result(context, result)
