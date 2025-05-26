from fastapi import APIRouter, Depends

from agentic.agents import sequence_agent
from agentic.schemas import SequenceRequest, SequenceResponse
from common.auth import verify_token

router = APIRouter(prefix="/agentic", tags=["Agentic"], dependencies=[Depends(verify_token)])


@router.post("/sequence", response_model=SequenceResponse, operation_id="agentic_sequence_create")
def create(request: SequenceRequest):
    return sequence_agent.main(request)
