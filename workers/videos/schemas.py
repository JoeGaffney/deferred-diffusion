from typing import Dict, Literal, Optional, TypeAlias, Union, get_args
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelNameLocal: TypeAlias = Literal["ltx-video", "wan-2"]

# External-only model names (convenience alias)
ModelNameExternal: TypeAlias = Literal[
    "runway-gen-3",
    "runway-gen-4",
    "runway-act-two",
    "runway-upscale",
    "runway-gen-4-aleph",
    "bytedance-seedance-1",
    "kwaivgi-kling-2",
    "minimax-hailuo-2",
    "google-veo-3",
]


# User facing choice
ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2",
    "runway-gen-3",
    "runway-gen-4",
    "runway-act-two",
    "runway-upscale",
    "runway-gen-4-aleph",
    "bytedance-seedance-1",
    "kwaivgi-kling-2",
    "minimax-hailuo-2",
    "google-veo-3",
]


class ModelInfo(BaseModel):
    description: str

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"### {model_name}\n\n"
        doc += f"{self.description}\n\n"
        doc += "\n"
        return doc


MODEL_META_LOCAL: Dict[ModelNameLocal, ModelInfo] = {
    "ltx-video": ModelInfo(
        description="Fast but more limted video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2": ModelInfo(
        description="Wan 2.2, best open-source video generation model. Good quality and motion coherence for a variety of scenes.",
    ),
}

MODEL_META_EXTERNAL: Dict[ModelNameExternal, ModelInfo] = {
    "runway-gen-3": ModelInfo(
        description="Runway's Gen-3 model okay for basic video generation tasks. Suitable for simple scenes and concepts.",
    ),
    "runway-gen-4": ModelInfo(
        description="Runway's latest Gen-4 showing its age in fast computer time. Good for a variety of video generation tasks with improved quality over Gen-3.",
    ),
    "runway-act-two": ModelInfo(
        description="Runway's Act Two model updates a video with reference image. Ideal for enhancing existing footage with new visual elements while maintaining original motion and style.",
    ),
    "runway-upscale": ModelInfo(
        description="Runway's Upscale model for high-quality video upscaling. Utilizes advanced techniques to enhance video resolution and detail.",
    ),
    "runway-gen-4-aleph": ModelInfo(
        description="Runway's Gen-4 Aleph model, takes in video input as well as images and can enhance or change the video. Or even generate new video content based on the input images and video. Ideal for creative video transformations and enhancements.",
    ),
    "bytedance-seedance-1": ModelInfo(
        description="ByteDance's Seedance-1 model, Pretty strong overall and quite fast. Good for a variety of video generation tasks with decent quality and speed.",
    ),
    "kwaivgi-kling-2": ModelInfo(
        description="Kling V2 model by kwaivgi, designed for high-quality video generation from text prompts. Known for its ability to create detailed and coherent video sequences.",
    ),
    "google-veo-3": ModelInfo(
        description="Google's VEO-3-Fast model, optimized for rapid video generation while maintaining good quality. Suitable for applications requiring quick turnaround times.",
    ),
    "minimax-hailuo-2": ModelInfo(
        description="Minimax's Hailuo-02 model, a versatile video generation model capable of producing high-quality videos from text prompts. Balances quality and performance effectively.",
    ),
}


def generate_model_docs():
    """Generate documentation about available video models"""
    docs = "# Generate videos using various diffusion models.\n\n"

    docs += "# Local Models\n\n"
    for model_name, model_info in sorted(MODEL_META_LOCAL.items()):
        docs += model_info.to_doc_format(model_name)

    docs += "# External Models\n\n"
    for model_name, model_info in sorted(MODEL_META_EXTERNAL.items()):
        docs += model_info.to_doc_format(model_name)

    return docs


class VideoRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Slow camera zoom in, 4k, high quality, cinematic, realistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    num_frames: int = 48
    seed: int = 42
    image: Optional[str] = Field(
        default=None,
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    image_last_frame: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string for the last frame",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    video: Optional[str] = Field(
        default=None,
        description="Optional Base64 video string for video input",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "video/*",
        },
    )

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


class VideoWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class VideoResponse(BaseModel):
    id: UUID
    status: str
    result: Optional[VideoWorkerResponse] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
                },
                "error_message": None,
            }
        }


class VideoCreateResponse(BaseModel):
    id: UUID
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
