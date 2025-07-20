from typing import Any, Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

# User facing choice
ModelName: TypeAlias = Literal[
    "sd-xl",
    "sd-3",
    "flux-1",
    "flux-kontext-1",
    "depth-anything-2",
    "segment-anything-2",
    "sd-x4-upscaler",
    "external-gpt-image-1",
    "external-runway-gen4-image",
    "external-flux-kontext",
    "external-flux-1-1",
]

# Family will map to the model pipeline and usually match the pyhon module name
ModelFamily: TypeAlias = Literal[
    "sdxl", "sd3", "flux", "openai", "runway", "sd_upscaler", "segment_anything", "depth_anything", "flux_kontext"
]


class ModelInfo(BaseModel):
    family: ModelFamily
    path: str
    external: bool
    inpainting_path: Optional[str] = None
    auto_divisor: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Divisor for image dimensions. If above 1, the image dimensions will be adjusted to be divisible by this value.",
    )
    controlnets: bool = Field(default=False, description="Whether the model supports ControlNets.")
    adapters: bool = Field(default=False, description="Whether the model supports IP Adapters.")
    inpainting: bool = Field(
        default=False,
        description="Whether the model supports inpainting. If true, the inpainting path will be used for inpainting tasks.",
    )
    description: Optional[str] = None

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"## {model_name}\n\n"
        doc += f"{self.description}\n"

        doc += f"- **Path:** `{self.path}`\n"

        if self.inpainting_path:
            doc += f"- **Inpainting Path:** `{self.inpainting_path}`\n"

        doc += f"- **Family:** {self.family}\n"
        doc += f"- **External API:** {'Yes' if self.external else 'No'}\n"
        doc += f"- **Auto Divisor:** {self.auto_divisor}\n"
        doc += f"- **ControlNets:** {'✓' if self.controlnets else '✗'}\n"
        doc += f"- **IP Adapters:** {'✓' if self.adapters else '✗'}\n"
        doc += f"- **Inpainting:** {'✓' if self.inpainting else '✗'}\n"

        doc += "\n"
        return doc


MODEL_CONFIG: Dict[ModelName, ModelInfo] = {
    "sd-xl": ModelInfo(
        family="sdxl",
        path="SG161222/RealVisXL_V4.0",
        inpainting_path="OzzyGT/RealVisXL_V4.0_inpainting",
        external=False,
        auto_divisor=8,
        controlnets=True,
        adapters=True,
        inpainting=True,
        description="Stable Diffusion XL variant supports the most control nets and IP adapters. It excels at generating high-quality, detailed images with complex prompts and multiple subjects.",
    ),
    "sd-3": ModelInfo(
        family="sd3",
        path="stabilityai/stable-diffusion-3.5-large",
        external=False,
        auto_divisor=16,
        controlnets=True,
        adapters=True,
        inpainting=True,
        description="Stable Diffusion 3.5 offers superior prompt understanding and composition. Excels at complex scenes, concept art, and handling multiple subjects with accurate interactions.",
    ),
    "flux-1": ModelInfo(
        family="flux",
        path="black-forest-labs/FLUX.1-dev",
        inpainting_path="black-forest-labs/FLUX.1-Fill-dev",
        external=False,
        auto_divisor=32,
        controlnets=True,
        adapters=True,
        inpainting=True,
        description="FLUX.1 delivers exceptional text rendering and coherent scene composition. Specializes in illustrations with text elements, UI mockups, and detailed technical imagery.",
    ),
    "flux-kontext-1": ModelInfo(
        family="flux_kontext",
        path="black-forest-labs/FLUX.1-Kontext-dev",
        external=False,
        description="FLUX Kontext model optimized for contextual understanding. Ideal for scene continuation, thematically consistent series, and context-aware image generation.",
    ),
    "depth-anything-2": ModelInfo(
        family="depth_anything",
        path="depth-anything/Depth-Anything-V2-Large-hf",
        external=False,
        description="Advanced depth estimation model. Creates high-quality depth maps from any image for 3D visualization, AR applications, and as input for ControlNet pipelines.",
    ),
    "segment-anything-2": ModelInfo(
        family="segment_anything",
        path="sam2.1_hiera_base_plus",
        external=False,
        description="State-of-the-art image segmentation model. Precisely identifies and segments objects, people, and features for compositing, editing, and analysis.",
    ),
    "sd-x4-upscaler": ModelInfo(
        family="sd_upscaler",
        path="stabilityai/stable-diffusion-x4-upscaler",
        external=False,
        description="Specialized 4x upscaling model that enhances image resolution while adding realistic details. Excellent for enlarging images without quality loss.",
    ),
    "external-gpt-image-1": ModelInfo(
        family="openai",
        path="gpt-image-1",
        external=True,
        adapters=True,
        inpainting=True,
        description="OpenAI's advanced image generation model with exceptional understanding of complex prompts. Excels at photorealistic imagery, accurate object rendering, and following detailed instructions.",
    ),
    "external-runway-gen4-image": ModelInfo(
        family="runway",
        path="gen4_image",
        external=True,
        adapters=True,
        description="Runway's Gen-4 image model delivering high-fidelity results with strong coherence. Particularly good at combining multiple references into a single, cohesive image.",
    ),
    "external-flux-kontext": ModelInfo(
        family="flux_kontext",
        path="black-forest-labs/flux-kontext-pro",
        external=True,
        description="Professional version of FLUX Kontext accessed via API. Offers superior contextual awareness for creating coherent image sequences and variations with consistent themes.",
    ),
    "external-flux-1-1": ModelInfo(
        family="flux",
        path="black-forest-labs/flux-1.1-pro",
        inpainting_path="black-forest-labs/flux-fill-pro",
        external=True,
        auto_divisor=32,
        inpainting=True,
        description="Pro version of FLUX 1.1 with enhanced capabilities. Excellent text rendering, sharp details, and consistent style. Ideal for professional illustrations and design work.",
    ),
}


def generate_model_docs():
    docs = "Generate images using various diffusion models.\n\n"

    # List all models alphabetically without family grouping
    docs += "# Available Models\n\n"

    for model_name, model_info in MODEL_CONFIG.items():
        docs += model_info.to_doc_format(model_name)

    # Add notes
    docs += "# Notes \n"
    docs += "- External models are processed through their respective APIs.\n"
    docs += "- Auto Divisor ensures image dimensions are divisible by the specified value.\n"
    docs += "- ControlNets allow for controlling structure and placement.\n"
    docs += "- IP Adapters guide the image prompt with images - for style and visual reference.\n"
    docs += "- Guidance Scale controls the strength of the prompt influence on the image generation. This can have a different effect with each model, but general rule of thumb is lower values produce more realism.\n"
    docs += "- If there is no image the models will do a text-to-image generation, if there is an image it will do an image-to-image generation.\n"
    docs += "- If a mask is provided and the model supports inpainting, it will use the mask to guide the inpainting process.\n"

    return docs


# External models are non-local models processed by external services
TaskName: TypeAlias = Literal["process_image", "process_image_external"]


class IpAdapterModelConfig(BaseModel):
    model: str = Field(
        description="The model name for the IP adapter.",
    )
    subfolder: str = Field(
        description="The subfolder where the IP adapter model is stored.",
    )
    weight_name: str = Field(
        description="The weight name for the IP adapter model.",
    )
    image_encoder: bool = Field(description="Whether to use the image encoder for the IP adapter model.")
    image_encoder_subfolder: str = Field(description="The subfolder where the image encoder model is stored.")


class ControlNetSchema(BaseModel):
    model: Literal[
        "pose",
        "depth",
        "canny",
    ]
    conditioning_scale: float = 0.5
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/png, image/jpg, image/jpeg",  # Or "image/*" if you accept multiple formats
        },
    )


class IpAdapterModel(BaseModel):
    model: Literal[
        "style",
        "style-plus",
        "face",
    ]
    scale: float = 0.5
    scale_layers: str = "all"
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    mask: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )


class ImageRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Detailed, 8k, photorealistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    height: int = 512
    width: int = 512
    num_inference_steps: int = 25
    seed: int = 42
    guidance_scale: float = 5.0
    strength: float = 0.5
    image: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    mask: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    ip_adapters: list[IpAdapterModel] = []
    controlnets: list[ControlNetSchema] = []

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_CONFIG[self.model].family

    @property
    def model_path(self) -> str:
        return MODEL_CONFIG[self.model].path

    @property
    def model_divisor(self) -> int:
        return MODEL_CONFIG[self.model].auto_divisor

    @property
    def model_path_inpainting(self) -> str:
        """Get the inpainting model path if available, otherwise normal."""
        inpainting_path = MODEL_CONFIG[self.model].inpainting_path
        if inpainting_path:
            return inpainting_path
        return self.model_path

    @property
    def external_model(self) -> bool:
        return MODEL_CONFIG[self.model].external

    @property
    def task_name(self) -> TaskName:
        return "process_image_external" if self.external_model else "process_image"

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return "cpu" if self.external_model else "gpu"


class ImageWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class ImageResponse(BaseModel):
    id: UUID
    status: str
    result: Optional[ImageWorkerResponse] = None
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


class ImageCreateResponse(BaseModel):
    id: UUID
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
