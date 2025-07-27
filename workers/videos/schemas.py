from typing import Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2-1",
    "external-runway-gen-3",
    "external-runway-gen-4",
    "external-runway-act-two",
    "external-runway-upscale",
]
ModelFamily: TypeAlias = Literal["ltx", "wan", "runway", "runway_act", "runway_upscale"]
TaskName: TypeAlias = Literal["process_video", "process_video_external"]


class ModelInfo(BaseModel):
    family: ModelFamily
    path: str
    external: bool
    description: Optional[str] = None

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"### {model_name}\n\n"

        if self.description:
            doc += f"{self.description}\n\n"

        doc += f"- **Path:** `{self.path}`\n"
        doc += f"- **Family:** {self.family}\n"
        doc += f"- **External API:** {'Yes' if self.external else 'No'}\n"

        doc += "\n"
        return doc


MODEL_CONFIG: Dict[ModelName, ModelInfo] = {
    "ltx-video": ModelInfo(
        family="ltx",
        path="Lightricks/LTX-Video-0.9.7-distilled",
        external=False,
        description="Fast but more limted video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2-1": ModelInfo(
        family="wan",
        path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        external=False,
        description="Powerful model with excellent temporal consistency. Specializes in maintaining subject identity and detailed motion from a single image.",
    ),
    "external-runway-gen-3": ModelInfo(
        family="runway",
        path="gen3a_turbo",
        external=True,
        description="Runway's Gen-3 model accessed via API. Great for dynamic scenes, camera movements, and natural motion. Good balance of quality and generation speed.",
    ),
    "external-runway-gen-4": ModelInfo(
        family="runway",
        path="gen4_turbo",
        external=True,
        description="Runway's latest Gen-4 model offering exceptional motion coherence and visual quality. Superior handling of complex animations and realistic physics.",
    ),
    "external-runway-act-two": ModelInfo(
        family="runway_act",
        path="act_two",
        external=True,
        description="Runway's Act Two model updates a video with reference image. Ideal for enhancing existing footage with new visual elements while maintaining original motion and style.",
    ),
    "external-runway-upscale": ModelInfo(
        family="runway_upscale",
        path="upscale_v1",
        external=True,
        description="Runway's Upscale model for high-quality video upscaling. Utilizes advanced techniques to enhance video resolution and detail.",
    ),
}


def generate_model_docs():
    """Generate documentation about available video models"""
    docs = "Generate videos using various diffusion models.\n\n"

    # List all models alphabetically
    docs += "# Available Models\n\n"

    for model_name, model_info in sorted(MODEL_CONFIG.items()):
        docs += model_info.to_doc_format(model_name)

    # Add notes section
    docs += "# Notes\n\n"
    docs += "- External models are processed through their respective APIs\n"
    docs += "- Local models run on your GPU infrastructure\n"
    docs += "- For best results with image-to-video generation:\n"
    docs += "  - Use high-quality input images\n"
    docs += "  - Keep prompts consistent with the visual content\n"

    return docs


class VideoRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Slow camera zoom in, 4k, high quality, cinematic, realistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
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
    def model_family(self) -> ModelFamily:
        return MODEL_CONFIG[self.model].family

    @property
    def model_path(self) -> str:
        return MODEL_CONFIG[self.model].path

    @property
    def external_model(self) -> bool:
        return MODEL_CONFIG[self.model].external

    @property
    def task_name(self) -> TaskName:
        """Determines the appropriate task name based on request characteristics."""
        if self.external_model:
            return "process_video_external"
        return "process_video"

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
