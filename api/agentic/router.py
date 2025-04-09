from fastapi import APIRouter

from agentic.agents import script_agent
from agentic.schemas import ScriptAgentRequest, ScriptAgentResponse

router = APIRouter(prefix="/agentic", tags=["Agentic"])


@router.post("/script_agent", response_model=ScriptAgentResponse, operation_id="script_agent")
def create(request: ScriptAgentRequest):
    return script_agent.main(request)
