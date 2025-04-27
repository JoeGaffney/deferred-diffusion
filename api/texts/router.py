from fastapi import APIRouter, HTTPException

from texts.context import TextContext
from texts.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main
from texts.schemas import TextRequest, TextResponse

router = APIRouter(prefix="/texts", tags=["Texts"])


@router.post("", response_model=TextResponse, operation_id="texts_create")
async def create(request: TextRequest):
    context = TextContext(request)

    main = None
    if request.model == "Qwen/Qwen2.5-VL-3B-Instruct":
        main = qwen_2_5_vl_instruct_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return TextResponse(response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", []))
