from common.memory import free_gpu_memory
from texts.context import TextContext
from texts.models.openai import main as openai_main
from texts.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main
from texts.schemas import TextRequest, TextWorkerResponse
from worker import celery_app


@celery_app.task(name="process_text")
def process_text(request_dict):
    free_gpu_memory()
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)

    result = None
    if request.model == "Qwen2.5-VL-3B-Instruct":
        result = qwen_2_5_vl_instruct_main(context)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    # Free GPU memory after processing
    free_gpu_memory(threshold_percent=1)
    return TextWorkerResponse(
        response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
    ).model_dump()


@celery_app.task(name="process_text_external")
def process_text_external(request_dict):
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)

    result = None
    if request.model == "gpt-4o-mini" or request.model == "gpt-4.1-mini":
        result = openai_main(context)
    else:
        raise ValueError(f"Unsupported model: {request.model}")

    return TextWorkerResponse(
        response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
    ).model_dump()
