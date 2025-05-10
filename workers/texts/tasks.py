from texts.context import TextContext
from texts.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main
from texts.schemas import TextRequest, TextResponse
from worker import celery_app  # Import from worker.py


@celery_app.task(name="process_text")
def process_text(request: TextRequest):
    context = TextContext(request)

    main = None
    if request.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        main = qwen_2_5_vl_instruct_main

    if not main:
        raise ValueError(f"Invalid model {request.model}")

    result = main(context)
    return TextResponse(response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", []))
