from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from common.context import Context
from text.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main

router = APIRouter(prefix="/text", tags=["Text"])


class TextRequest(BaseModel):
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: int = 48
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    model: str = "qwen_2_5_vl_instruct"
    messages: list = []


class TextResponse(BaseModel):
    response: str
    chain_of_thought: list


@router.post("/", response_model=TextResponse)
def create(request: TextRequest):
    context = Context(
        negative_prompt=request.negative_prompt,
        num_frames=request.num_frames,
        prompt=request.prompt,
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
