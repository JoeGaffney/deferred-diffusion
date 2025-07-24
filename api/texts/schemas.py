from typing import Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from celery.states import ALL_STATES
from pydantic import BaseModel, Field, RootModel

ModelName: TypeAlias = Literal["qwen-2-5", "external-gpt-4", "external-gpt-4-1"]
ModelFamily: TypeAlias = Literal["qwen", "openai"]
TaskName: TypeAlias = Literal["process_text", "process_text_external"]


class TextContent(BaseModel):
    type: str = "text"
    text: str


class MessageContent(BaseModel):
    type: str
    text: str = ""


class MessageItem(BaseModel):
    role: str
    content: List[MessageContent]


class TextRequest(BaseModel):
    temperature: float = 0.7
    seed: int = 42
    model: ModelName = Field(description="model", default="qwen-2-5")
    messages: list[MessageItem] = Field(description="List of messages", default=[])
    images: List[str] = Field(description="Image references", default=[])
    videos: List[str] = Field(description="Video references", default=[])

    @property
    def model_family(self) -> ModelFamily:
        mapping: Dict[ModelName, ModelFamily] = {
            "qwen-2-5": "qwen",
            "external-gpt-4": "openai",
            "external-gpt-4-1": "openai",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model family defined for model '{self.model}'")

    @property
    def model_path(self) -> str:
        mapping: Dict[ModelName, str] = {
            "qwen-2-5": "Qwen/Qwen2.5-VL-3B-Instruct",
            "external-gpt-4": "gpt-4o-mini",
            "external-gpt-4-1": "gpt-4.1-mini",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model path defined for model '{self.model}'")

    @property
    def external_model(self) -> bool:
        external_models = ["openai"]
        return self.model_family in external_models

    @property
    def task_name(self) -> TaskName:
        """Determines the appropriate task name based on request characteristics."""
        if self.external_model:
            return "process_text_external"
        return "process_text"

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return "cpu" if self.external_model else "gpu"


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
