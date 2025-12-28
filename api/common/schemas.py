from enum import Enum
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, Field

# NOTE Status values should align with Celery task states
# from celery.states import ALL_STATES


# NOTE need to use Enum to avoid duplicate in generated clients
class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"
    REJECTED = "REJECTED"
    IGNORED = "IGNORED"

    def __str__(self) -> str:
        return str(self.value)


class DeleteResponse(BaseModel):
    id: UUID = Field(description="ID of the task")
    status: TaskStatus = Field(description="Status of the task after deletion attempt")
    message: str = Field(description="Additional information about the deletion result")


Provider: TypeAlias = Literal["local", "openai", "replicate"]


class Identity(BaseModel):
    user_id: str
    machine_id: str
    client_ip: str
    key_name: str
    key_id: str


class APIKeyPublic(BaseModel):
    key_id: str
    name: str
    created_at: str
