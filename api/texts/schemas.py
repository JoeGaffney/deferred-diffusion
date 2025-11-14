from typing import Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from common.schemas import TaskStatus

ModelName: TypeAlias = Literal["qwen-2", "gpt-4o", "gpt-4", "gpt-5"]
Provider: TypeAlias = Literal["local", "openai"]


class TextsModelInfo(BaseModel):
    provider: Provider = Field(description="Source/provider identifier")
    external: bool = Field(description="True if the model is invoked via an external API")
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"


MODEL_META: Dict[ModelName, TextsModelInfo] = {
    "qwen-2": TextsModelInfo(
        provider="local",
        external=False,
        description="Qwen-2 is a high-performance language model optimized for text generation and conversation. Excels at reasoning, creative writing, and multi-turn conversations.",
    ),
    "gpt-4o": TextsModelInfo(
        provider="openai",
        external=True,
        description="OpenAI's GPT-4o model with enhanced multimodal capabilities. (mini variant)",
    ),
    "gpt-4": TextsModelInfo(
        provider="openai",
        external=True,
        description="OpenAI's GPT-4 model with advanced reasoning capabilities. (4.1 mini variant)",
    ),
    "gpt-5": TextsModelInfo(
        provider="openai",
        external=True,
        description="OpenAI's latest GPT-5 model with cutting-edge performance across all text generation tasks. (mini variant)",
    ),
}


def generate_model_docs():
    header = (
        "# Text Models\n"
        "External models proxy to provider APIs; local models run on your GPU.\n\n"
        "| Model | Provider | External | Queue | Description |\n"
        "|-------|----------|:--------:|:-----:|-------------|\n"
    )
    rows = []
    for name, meta in MODEL_META.items():
        rows.append(
            f"| {name} | {meta.provider} | {'Yes' if meta.external else 'No'} | {meta.queue} | {meta.description or ''} |"
        )
    return header + "\n".join(rows) + "\n"


class TextRequest(BaseModel):
    model: ModelName = Field(description="model", default="qwen-2")
    prompt: str = Field(description="Prompt text", default="")
    system_prompt: str = Field(
        description="System prompt",
        default=(
            "You are a helpful AI assistant specialized in visual effects, filmmaking, image generation, and creative workflows. "
            "You excel at analyzing images and videos, describing visual content, and generating detailed prompts for AI image/video generation models. "
            "When given images or videos, provide clear, detailed descriptions focusing on visual elements, composition, lighting, style, and technical aspects. "
            "When asked to create prompts, generate specific, detailed descriptions that would work well with AI generation models like Flux, Runway, or Stable Diffusion. "
            "Provide concise, actionable responses optimized for creative production pipelines. "
            "Do not ask for clarification - provide the best possible response based on the given input. "
            "Do not describe what you are doing or ask follow up questions."
            "Use any images or videos provided in the conversation to inform your responses."
        ),
    )
    images: List[str] = Field(description="Image references", default=[])
    videos: List[str] = Field(description="Video references", default=[])

    @property
    def meta(self) -> TextsModelInfo:
        return MODEL_META[self.model]

    @property
    def external_model(self) -> bool:
        return self.meta.external

    @property
    def task_name(self) -> ModelName:
        return self.model

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return self.meta.queue


class TextWorkerResponse(BaseModel):
    response: str


class TextResponse(BaseModel):
    id: UUID
    status: TaskStatus
    result: Optional[TextWorkerResponse] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "response": "This is a response from the model",
                },
                "error_message": None,
            }
        }
    )


class TextCreateResponse(BaseModel):
    id: UUID
    status: TaskStatus
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
    )


class TextModelsResponse(BaseModel):
    models: Dict[ModelName, TextsModelInfo]
