from fastapi import APIRouter

from agentic.agents import sequence_agent
from agentic.schemas import SequenceAgentRequest, SequenceAgentResponse

router = APIRouter(prefix="/agentic", tags=["Agentic"])


@router.post("/sequence_agent", response_model=SequenceAgentResponse, operation_id="sequence_agent")
def create(request: SequenceAgentRequest):
    return sequence_agent.main(request)
