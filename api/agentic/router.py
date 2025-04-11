from fastapi import APIRouter

from agentic.agents import sequence_agent
from agentic.schemas import SequenceRequest, SequenceResponse

router = APIRouter(prefix="/agentic", tags=["Agentic"])


@router.post("/sequence_agent", response_model=SequenceResponse, operation_id="sequence_agent")
def create(request: SequenceRequest):
    return sequence_agent.main(request)
