from typing import Dict, Literal, Optional, TypeAlias, Union, get_args
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelNameLocal: TypeAlias = Literal["ltx-video", "wan-2-1", "wan-2-2"]

# External-only model names (convenience alias)
ModelNameExternal: TypeAlias = Literal[
    "runway-gen-3",
    "runway-gen-4",
    "runway-act-two",
    "runway-upscale",
    "runway-gen-4-aleph",
    "bytedance-seedance-1",
    "kwaivgi-kling-2-1",
    "minimax-hailuo-2",
    "google-veo-3",
]


# User facing choice
ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2-1",
    "wan-2-2",
    "runway-gen-3",
    "runway-gen-4",
    "runway-act-two",
    "runway-upscale",
    "runway-gen-4-aleph",
    "bytedance-seedance-1",
    "kwaivgi-kling-2-1",
    "minimax-hailuo-2",
    "google-veo-3",
]


class ModelInfo(BaseModel):
    path: str
    description: Optional[str] = None

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"### {model_name}\n\n"

        if self.description:
            doc += f"{self.description}\n\n"

        doc += f"- **Path:** `{self.path}`\n"

        doc += "\n"
        return doc


MODEL_META_LOCAL: Dict[ModelNameLocal, ModelInfo] = {
    "ltx-video": ModelInfo(
        path="Lightricks/LTX-Video-0.9.7-distilled",
        description="Fast but more limted video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2-1": ModelInfo(
        path="Wan-AI/Wan2.1-I2V-A14B-Diffusers",
        description="Previous version of Wan 2.2, still strong performance but slightly less temporal consistency.",
    ),
    "wan-2-2": ModelInfo(
        path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        description="Powerful model with excellent temporal consistency. Specializes in maintaining subject identity and detailed motion from a single image.",
    ),
}

MODEL_META_EXTERNAL: Dict[ModelNameExternal, ModelInfo] = {
    "runway-gen-3": ModelInfo(
        path="gen3a_turbo",
        description="Runway's Gen-3 model accessed via API. Great for dynamic scenes, camera movements, and natural motion. Good balance of quality and generation speed.",
    ),
    "runway-gen-4": ModelInfo(
        path="gen4_turbo",
        description="Runway's latest Gen-4 model offering exceptional motion coherence and visual quality. Superior handling of complex animations and realistic physics.",
    ),
    "runway-act-two": ModelInfo(
        path="act_two",
        description="Runway's Act Two model updates a video with reference image. Ideal for enhancing existing footage with new visual elements while maintaining original motion and style.",
    ),
    "runway-upscale": ModelInfo(
        path="upscale_v1",
        description="Runway's Upscale model for high-quality video upscaling. Utilizes advanced techniques to enhance video resolution and detail.",
    ),
    "runway-gen-4-aleph": ModelInfo(
        path="gen4_aleph",
        description="Runway's Gen-4 Aleph model, takes in video input as well as images and can enhance or change the video. Or even generate new video content based on the input images and video. Ideal for creative video transformations and enhancements.",
    ),
    "bytedance-seedance-1": ModelInfo(
        path="bytedance/seedance-1-lite",
        description="ByteDance's Seedance-1-Lite model, a lighter version of their Seedance-1 model. Good for generating dance videos from text prompts with lower computational requirements.",
    ),
    "kwaivgi-kling-2-1": ModelInfo(
        path="kwaivgi/kling-v2.1",
        description="Kling V2.1 model by kwaivgi, designed for high-quality video generation from text prompts. Known for its ability to create detailed and coherent video sequences.",
    ),
    "google-veo-3": ModelInfo(
        path="google/veo-3-fast",
        description="Google's VEO-3-Fast model, optimized for rapid video generation while maintaining good quality. Suitable for applications requiring quick turnaround times.",
    ),
    "minimax-hailuo-2": ModelInfo(
        path="minimax/hailuo-02",
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
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes",
        description="Negative prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    guidance_scale: float = 5.0
    num_frames: int = 48
    num_inference_steps: int = 25
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
