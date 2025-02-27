from common.context import Context
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from text.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main

router = APIRouter(prefix="/text", tags=["Text"])


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: str = "qwen_2_5_vl_instruct"
    messages: list


class TextResponse(BaseModel):
    response: str
    chain_of_thought: list


@router.post("/", response_model=TextResponse)
def create(request: TextRequest):
    context = Context(
        seed=request.seed,
        model=request.model,
        messages=request.messages,
    )

    main = None
    if request.model == "qwen_2_5_vl_instruct":
        main = qwen_2_5_vl_instruct_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return TextResponse(response=result.get("response", ""), chain_of_thought=result.get("chain_of_thought", []))
