from texts.context import TextContext
from texts.schemas import ModelName, TextRequest, TextWorkerResponse
from worker import celery_app


def process_result(context, result):
    """Process the result from text model and return standardized response."""
    if isinstance(result, dict):
        return TextWorkerResponse(
            response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
        ).model_dump()
    raise ValueError("Text generation failed")


# Helper to validate request and build context to avoid duplication across tasks
def validate_request_and_context(request_dict):
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)
    return context


def typed_task(name: ModelName, queue: str):
    return celery_app.task(name=name, queue=queue)


@typed_task(name="qwen-2", queue="gpu")
def qwen_2(request_dict):
    from texts.models.qwen import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-4o", queue="cpu")
def gpt_4o(request_dict):
    from texts.models.openai import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-4", queue="cpu")
def gpt_4(request_dict):
    from texts.models.openai import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)


@typed_task(name="gpt-5", queue="cpu")
def gpt_5(request_dict):
    from texts.models.openai import main

    context = validate_request_and_context(request_dict)
    result = main(context)
    return process_result(context, result)
