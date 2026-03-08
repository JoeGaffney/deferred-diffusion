from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    output_files: List[str] = Field(
        default_factory=list, description="List of file paths or signed URLs (images, videos, audio)"
    )
    output_text: List[str] = Field(
        default_factory=list, description="Array of generated text outputs (LLMs, transcriptions)"
    )
    logs: List[str] = Field(default_factory=list, description="Execution logs or warnings")
    error: Optional[str] = Field(None, description="Error message if failed")


class BaseTaskParams(BaseModel):
    pass


class BaseTaskParams(BaseModel):
    pass


class BaseTaskParams(BaseModel):
    pass


class BaseTaskParams(BaseModel):
    pass


class BaseTaskParams(BaseModel):
    pass
