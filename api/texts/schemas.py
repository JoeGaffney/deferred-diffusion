from typing import Dict, List, Literal, Optional, TypeAlias, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ModelNameLocal: TypeAlias = Literal["qwen-2"]

ModelNameExternal: TypeAlias = Literal["gpt-4o", "gpt-4", "gpt-5"]

ModelName: TypeAlias = Literal["qwen-2", "gpt-4o", "gpt-4", "gpt-5"]


class ModelInfo(BaseModel):
    description: Optional[str] = None

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"## {model_name}\n\n"
        doc += f"{self.description}\n"
        doc += "\n"
        return doc


MODEL_META_LOCAL: Dict[ModelNameLocal, ModelInfo] = {
    "qwen-2": ModelInfo(
        description="Qwen-2 is a high-performance language model optimized for text generation and conversation. Excels at reasoning, creative writing, and multi-turn conversations.",
    ),
}

MODEL_META_EXTERNAL: Dict[ModelNameExternal, ModelInfo] = {
    "gpt-4o": ModelInfo(
        description="OpenAI's GPT-4o model with enhanced multimodal capabilities. (mini variant)",
    ),
    "gpt-4": ModelInfo(
        description="OpenAI's GPT-4 model with advanced reasoning capabilities. (4.1 mini variant)",
    ),
    "gpt-5": ModelInfo(
        description="OpenAI's latest GPT-5 model with cutting-edge performance across all text generation tasks. (mini variant)",
    ),
}


def generate_model_docs():
    docs = """ # Generate text using various language models.
- External models are processed through their respective APIs.
- Temperature controls randomness: lower values (0.1-0.3) for focused responses, higher values (0.7-1.0) for creative output.
- Messages support conversation context with role-based structure (system, user, assistant).
- Image and video references can be included for multimodal processing (where supported).
"""
    docs += "# Local Models\n\n"
    for model_name, model_info in MODEL_META_LOCAL.items():
        docs += model_info.to_doc_format(model_name)

    docs += "# External Models\n\n"
    for model_name, model_info in MODEL_META_EXTERNAL.items():
        docs += model_info.to_doc_format(model_name)
    return docs


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
    model: ModelName = Field(description="model", default="qwen-2")
    temperature: float = 0.7
    seed: int = 42
    messages: list[MessageItem] = Field(description="List of messages", default=[])
    images: List[str] = Field(description="Image references", default=[])
    videos: List[str] = Field(description="Video references", default=[])

    @property
    def external_model(self) -> bool:
        _MODEL_EXTERNAL_VALUES = tuple(get_args(ModelNameExternal))
        return self.model in _MODEL_EXTERNAL_VALUES

    @property
    def task_name(self) -> ModelName:
        return self.model

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
    model_config = ConfigDict(
        json_schema_extra={
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
    )


class TextCreateResponse(BaseModel):
    id: UUID
    status: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
    )
