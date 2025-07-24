from common.memory import free_gpu_memory
from texts.context import TextContext
from texts.models.openai import main as openai_main
from texts.models.qwen import main as qwen_main
from texts.schemas import TextRequest, TextWorkerResponse
from worker import celery_app


@celery_app.task(name="process_text", queue="gpu")
def process_text(request_dict):
    free_gpu_memory()
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)
    family = context.data.model_family

    result = None
    if family == "qwen":
        result = qwen_main(context)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    # Free GPU memory after processing
    free_gpu_memory(threshold_percent=1)
    return TextWorkerResponse(
        response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
    ).model_dump()


@celery_app.task(name="process_text_external", queue="cpu")
def process_text_external(request_dict):
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)
    family = context.data.model_family

    result = None
    if family == "openai":
        result = openai_main(context)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    return TextWorkerResponse(
        response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
    ).model_dump()
