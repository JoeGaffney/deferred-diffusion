from fastapi import APIRouter

from agentic.agents import sequence_agent
from agentic.schemas import SequenceRequest, SequenceResponse

router = APIRouter(prefix="/agentic", tags=["Agentic"])


@router.post("/sequence", response_model=SequenceResponse, operation_id="agentic_sequence_create")
def create(request: SequenceRequest):
    return sequence_agent.main(request)
