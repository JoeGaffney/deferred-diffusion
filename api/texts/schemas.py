from enum import Enum
from typing import Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from common.schemas import Provider, TaskStatus

ModelName: TypeAlias = Literal["qwen-2", "gpt-4o", "gpt-4", "gpt-5"]


class TextsModelInfo(BaseModel):
    provider: Provider = Field(description="Source/provider identifier")
    external: bool = Field(description="True if the model is invoked via an external API")
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"


class SystemPrompt(str, Enum):
    NONE = "NONE"
    BASE = "BASE"
    IMAGE_OPTIMIZER = "IMAGE_OPTIMIZER"
    VIDEO_OPTIMIZER = "VIDEO_OPTIMIZER"
    VIDEO_TRANSITION = "VIDEO_TRANSITION"


SYSTEM_PROMPT_TEXT = {
    SystemPrompt.NONE: "",
    SystemPrompt.BASE: (
        "You analyze visual content. Be direct and specific. "
        "Answer based on provided images/videos without asking for clarification."
        "Use any reference images provided to inform the prompt."
    ),
    SystemPrompt.IMAGE_OPTIMIZER: (
        "Optimize & enhance the user's prompt for image generation. "
        "Describe: subject, setting, style, lighting, composition. "
        "Be specific and concise. Default to photorealism unless requested otherwise. "
        "Use any reference images provided to inform the prompt. But you don't need to describe the images again. "
        "If only one reference image is provided, use it as the basis for the prompt. Likely the user wants a variation or edit operation of that image."
        "Keep it brief and concrete. No filler words or quality descriptors unless essential. "
    ),
    SystemPrompt.VIDEO_OPTIMIZER: (
        "Optimize & enhance the user's prompt for video generation. "
        "Describe: action, camera movement, environment, subject details. "
        "Be specific about motion and changes. Default to photorealism unless requested otherwise. "
        "If a reference image is provided use is as the starting point and frame for the video. "
        "Use any reference images provided to inform the prompt. But you don't need to describe the images again. "
        "Keep it brief and concrete. No filler words or quality descriptors unless essential. "
        "Don't put time markers just describe the video as a whole it is only one shot. "
    ),
    SystemPrompt.VIDEO_TRANSITION: (
        "Optimize & enhance the user's prompt for video start frame end frame video generation. "
        "Describe the transition between the two provided images. "
        "State what changes from start to end. Be direct and specific. "
        "Focus on: subject transformation, camera movement, environment changes, lighting shifts. "
        "Keep it brief and concrete. No filler words or quality descriptors unless essential. "
        "Don't put time markers just describe the video as a whole. "
    ),
}


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
    system_prompt: SystemPrompt = Field(
        description=(
            "System prompt type. Options:\n"
            "NONE: Will use the model's default behavior.\n\n"
            "BASE: " + SYSTEM_PROMPT_TEXT[SystemPrompt.BASE] + "\n\n"
            "IMAGE_OPTIMIZER: " + SYSTEM_PROMPT_TEXT[SystemPrompt.IMAGE_OPTIMIZER] + "\n\n"
            "VIDEO_OPTIMIZER: " + SYSTEM_PROMPT_TEXT[SystemPrompt.VIDEO_OPTIMIZER] + "\n\n"
            "VIDEO_TRANSITION: " + SYSTEM_PROMPT_TEXT[SystemPrompt.VIDEO_TRANSITION] + "\n"
        ),
        default=SystemPrompt.BASE,
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
    def task_name(self) -> str:
        return f"texts.{self.model}"

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return self.meta.queue

    @property
    def full_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEXT[self.system_prompt]


class TextWorkerResponse(BaseModel):
    response: str


class TextResponse(BaseModel):
    id: UUID
    status: TaskStatus
    output: str = ""
    error_message: Optional[str] = None
    logs: List[str] = []
    task_info: dict = Field(default_factory=dict)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "output": "This is the generated text response from the model.",
                "error_message": None,
                "logs": ["Processing..."],
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
