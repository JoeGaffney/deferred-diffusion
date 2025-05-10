from fastapi import APIRouter, HTTPException

from texts.schemas import TextRequest, TextResponse
from utils.utils import poll_task_until_complete
from worker import celery_app

router = APIRouter(prefix="/texts", tags=["Texts"])


@router.post("", response_model=TextResponse, operation_id="texts_create")
async def create(request: TextRequest):

    task_id = celery_app.send_task("process_text", args=request.model_dump())
    task_result = await poll_task_until_complete(task_id)

    try:
        return TextResponse(
            response=task_result.result.get("response", ""),
            chain_of_thought=task_result.result.get("chain_of_thought", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")
