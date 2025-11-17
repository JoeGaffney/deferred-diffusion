from texts.context import TextContext
from texts.schemas import ModelName, TextRequest, TextWorkerResponse
from worker import celery_app


def process_result(context, result: str):
    """Process the result from text model and return standardized response."""
    return TextWorkerResponse(response=result).model_dump()


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=name, queue=queue)


@typed_task(name="qwen-2", queue="gpu")
def qwen_2(request_dict):
    from texts.local.qwen_2 import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-4o", queue="cpu")
def gpt_4o(request_dict):
    from texts.external.openai_gpt import main_gpt_4o as main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-4", queue="cpu")
def gpt_4(request_dict):
    from texts.external.openai_gpt import main_gpt_4 as main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-5", queue="cpu")
def gpt_5(request_dict):
    from texts.external.openai_gpt import main_gpt_5 as main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
