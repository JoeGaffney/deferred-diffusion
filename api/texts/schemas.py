from typing import List, Optional
from uuid import UUID

from celery.states import ALL_STATES
from pydantic import Base64Bytes, BaseModel, Field


class TextWorkerResponse(BaseModel):
    response: str
    chain_of_thought: list


class TextResponse(BaseModel):
    id: UUID
    status: str
    result: Optional[TextWorkerResponse] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "response": "This is a response from the model",
                    "chain_of_thought": ["Step 1", "Step 2", "Conclusion"],
                },
                "error_message": None,
            }
        }


class TextCreateResponse(BaseModel):
    id: UUID
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    messages: list[dict] = Field(description="List of messages", default=[])
    images: List[str] = Field(description="Image references", default=[])
    videos: List[str] = Field(description="Video references", default=[])
