from fastapi import APIRouter, HTTPException
from text.context import TextContext
from text.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main
from text.schemas import TextRequest, TextResponse

router = APIRouter(prefix="/text", tags=["Text"])


@router.post("/", response_model=TextResponse)
def create(request: TextRequest):
    context = TextContext(request)

    main = None
    if request.model == "qwen_2_5_vl_instruct":
        main = qwen_2_5_vl_instruct_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return TextResponse(response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", []))
