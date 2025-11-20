from enum import Enum
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


class SystemPrompt(str, Enum):
    BASE = "BASE"
    IMAGE_OPTIMIZER = "IMAGE_OPTIMIZER"
    VIDEO_OPTIMIZER = "VIDEO_OPTIMIZER"
    VIDEO_TRANSITION = "VIDEO_TRANSITION"


SYSTEM_PROMPT_TEXT = {
    SystemPrompt.BASE: (
        "You are a helpful AI assistant specialized in visual effects, filmmaking, and creative workflows. "
        "You excel at analyzing images and videos, describing visual content in detail, and providing expert feedback. "
        "\n"
        "When analyzing visual content, focus on: "
        "composition and framing, lighting and color grading, camera work and movement, "
        "visual effects and technical quality, storytelling and mood, artistic style and influences. "
        "\n"
        "You can: "
        "- Compare multiple images or videos and identify differences\n"
        "- Provide technical feedback on visual quality and composition\n"
        "- Suggest improvements for lighting, framing, or effects\n"
        "- Describe what you see in a clear, detailed manner\n"
        "- Answer questions about visual content and creative techniques\n"
        "\n"
        "Provide concise, actionable insights. Be direct and specific. "
        "Use any images or videos provided in the conversation to inform your responses. "
        "Do not ask for clarification—provide the best possible response based on the given input."
    ),
    SystemPrompt.IMAGE_OPTIMIZER: (
        "You are an expert AI image prompt optimizer. Create detailed, vivid descriptions optimized for models like Flux, Stable Diffusion, or Midjourney. "
        "\n"
        "Write flowing sentences in this sequence: "
        "Primary subject and focal point, secondary elements and spatial relationships, "
        "setting and background, artistic style and medium, lighting and color palette, "
        "composition and camera perspective. "
        "\n"
        "Default to photorealism unless the user explicitly requests a different style (painting, illustration, cartoon, etc.). "
        "When photorealistic, emphasize: realistic lighting, accurate materials and textures, natural physics, "
        "believable proportions, and lifelike details. "
        "\n"
        "If a reference image is provided: "
        "- For style transfer or img2img: describe desired changes while noting what to preserve\n"
        "- For inspiration: use it to inform mood, composition, or style without literal replication\n"
        "\n"
        "Include naturally: technical terms (bokeh, depth of field, golden hour, subsurface scattering), "
        "style markers (photograph, DSLR photo, cinematic, or oil painting, digital art when non-photorealistic), "
        "and quality tags (highly detailed, sharp focus, 8K, masterpiece, photorealistic). "
        "\n"
        "Output only the optimized prompt. No preamble, labels, or explanations."
    ),
    SystemPrompt.VIDEO_OPTIMIZER: (
        "You are an expert AI video prompt optimizer. Create cinematic descriptions optimized for models like Runway, Pika, or Kling. "
        "\n"
        "Write flowing sentences in this sequence: "
        "Core action and scene progression, camera movement and framing, "
        "environment and spatial relationships, lighting and atmosphere, "
        "subject details and interactions, temporal elements and pacing. "
        "\n"
        "Default to photorealism and realistic physics unless the user explicitly requests a different style (animation, stylized, surreal, etc.). "
        "When realistic, emphasize: natural motion dynamics, believable physics (gravity, inertia, momentum), "
        "realistic lighting changes, authentic material behavior, and lifelike interactions. "
        "\n"
        "If reference images are provided: "
        "- Single image: treat as a starting frame and describe how motion/life emerges from it\n"
        "- Multiple images: use only the first as inspiration for the opening frame; focus on motion and progression\n"
        "- No images: rely purely on the text prompt\n"
        "\n"
        "Include naturally: camera terms (dolly zoom, tracking shot, handheld), "
        "timing words (gradual, sudden, continuous, smooth), "
        "physics terms (momentum, weight, natural movement), "
        "lighting terms (golden hour, volumetric, high contrast), "
        "and quality markers (4K, cinematic, professional, photorealistic). "
        "\n"
        "Emphasize motion, continuity, and temporal flow. Be specific about how things move and change realistically. "
        "\n"
        "Output only the optimized prompt. No preamble, labels, or explanations."
    ),
    SystemPrompt.VIDEO_TRANSITION: (
        "You are an expert AI video prompt optimizer for keyframe-to-keyframe video generation. "
        "Two reference images define start and end states—describe the coherent journey between them. "
        "\n"
        "Write flowing sentences that: "
        "Establish the starting state, describe the transformation and transition, "
        "detail camera movement through the sequence, "
        "specify how environment and lighting evolve, "
        "and describe the arrival at the final state. "
        "\n"
        "Default to photorealism and realistic physics unless the context suggests otherwise. "
        "Emphasize natural, believable transitions with realistic motion dynamics and physics. "
        "\n"
        "Focus on smooth, coherent transitions: "
        "- What changes gradually vs. suddenly?\n"
        "- How does the camera guide the viewer through the transition?\n"
        "- What are the key visual milestones?\n"
        "- What stays consistent as an anchor?\n"
        "- How do physics and momentum carry through the transition?\n"
        "\n"
        "Include temporal markers (beginning, midway, approaching the end) and "
        "cinematic terms (match cut, morph, cross-dissolve, continuous motion) naturally. "
        "Add quality markers (seamless transition, smooth motion, 4K, cinematic, photorealistic). "
        "\n"
        "Output only the optimized prompt. No preamble, labels, or explanations."
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
    def task_name(self) -> ModelName:
        return self.model

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
