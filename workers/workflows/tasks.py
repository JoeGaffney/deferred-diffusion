from typing import List

from common.logger import get_task_logs
from worker import celery_app
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowOutput, WorkflowRequest, WorkflowWorkerResponse


def process_result(context: WorkflowContext, result: List[WorkflowOutput]):
    for output in result:
        context.validate_and_save_output(output)

    return WorkflowWorkerResponse(outputs=result, logs=get_task_logs()).model_dump()


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = WorkflowRequest.model_validate(request_dict)
    context = WorkflowContext(request)
    return context


@celery_app.task(name="workflows.comfy-workflow", queue="comfy")
def comfy_workflow(request_dict):
    from workflows.comfy.comfy_workflow import main

    context = validate_request_and_context(request_dict)
    result = main(context)

    return process_result(context, result)
