from texts.context import TextContext
from texts.models.openai import main as openai_main
from texts.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main
from texts.schemas import TextRequest, TextWorkerResponse
from utils.utils import free_gpu_memory
from worker import celery_app


@celery_app.task(name="process_text")
def process_text(request_dict):
    free_gpu_memory()
    request = TextRequest.model_validate(request_dict)
    context = TextContext(request)

    main = None
    if request.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        main = qwen_2_5_vl_instruct_main
    elif request.model == "gpt-4o-mini":
        main = openai_main
    elif request.model == "gpt-4.1-mini":
        main = openai_main

    if not main:
        raise ValueError(f"Invalid model {request.model}")

    result = main(context)
    return TextWorkerResponse(
        response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", [])
    ).model_dump()
