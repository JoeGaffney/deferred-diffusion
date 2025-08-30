from typing import Any, Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

# User facing choice
ModelName: TypeAlias = Literal[
    "sd-xl",
    "sd-3",
    "flux-1",
    "flux-1-krea",
    "flux-kontext-1",
    "qwen-image",
    "depth-anything-2",
    "segment-anything-2",
    "real-esrgan-x4",
    "gpt-image-1",
    "runway-gen4-image",
    "flux-kontext-1-pro",
    "flux-1-1-pro",
    "topazlabs-upscale",
]

# Family will map to the model pipeline and usually match the pyhon module name
ModelFamily: TypeAlias = Literal[
    "sdxl",
    "sd3",
    "flux",
    "qwen",
    "openai",
    "runway",
    "real_esrgan",
    "segment_anything",
    "depth_anything",
    "flux_kontext",
    "topazlabs",
]


class ModelInfo(BaseModel):
    family: ModelFamily
    path: str
    external: bool
    inpainting_path: Optional[str] = None
    image_to_image_path: Optional[str] = None
    references: bool = Field(default=False, description="Supports image references as input")

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
        doc += f"- **References:** {'✓' if self.references else '✗'}\n"

        doc += "\n"
        return doc


MODEL_CONFIG: Dict[ModelName, ModelInfo] = {
    "sd-xl": ModelInfo(
        family="sdxl",
        path="SG161222/RealVisXL_V4.0",
        inpainting_path="OzzyGT/RealVisXL_V4.0_inpainting",
        external=False,
        references=True,
        description="Stable Diffusion XL variant supports the most conteol nets and IP adapters. It excels at generating high-quality, detailed images with complex prompts and multiple subjects.",
    ),
    "sd-3": ModelInfo(
        family="sd3",
        path="stabilityai/stable-diffusion-3.5-large",
        external=False,
        description="Stable Diffusion 3.5 offers superior prompt understanding and composition. Excels at complex scenes, concept art, and handling multiple subjects with accurate interactions.",
    ),
    "flux-1": ModelInfo(
        family="flux",
        path="black-forest-labs/FLUX.1-dev",
        inpainting_path="black-forest-labs/FLUX.1-Fill-dev",
        external=False,
        references=True,
        description="FLUX.1 delivers exceptional text rendering and coherent scene composition. Specializes in illustrations with text elements, UI mockups, and detailed technical imagery.",
    ),
    "flux-1-krea": ModelInfo(
        family="flux",
        path="black-forest-labs/FLUX.1-Krea-dev",
        external=False,
        references=True,
        description="FLUX Krea model is flux-1 dev trained with opinions from krea for more photorealistic results.",
    ),
    "flux-kontext-1": ModelInfo(
        family="flux_kontext",
        path="black-forest-labs/FLUX.1-Kontext-dev",
        external=False,
        description="FLUX Kontext model optimized for contextual understanding. Ideal for scene continuation, thematically consistent series, and context-aware image generation.",
    ),
    "qwen-image": ModelInfo(
        family="qwen",
        path="ovedrive/qwen-image-4bit",  # path="Qwen/Qwen-Image",
        image_to_image_path="ovedrive/qwen-image-edit-4bit",
        external=False,
        description="Qwen model specializes in generating high-quality images from textual descriptions. It excels at understanding nuanced prompts and delivering detailed visuals.",
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
    "real-esrgan-x4": ModelInfo(
        family="real_esrgan",
        path="weights/RealESRGAN_x4plus.pth",
        external=False,
        description="x4 upscaler model based on Real-ESRGAN. Ideal for enhancing image resolution while preserving details and textures. Great for upscaling low-resolution images.",
    ),
    "gpt-image-1": ModelInfo(
        family="openai",
        path="gpt-image-1",
        external=True,
        references=True,
        description="OpenAI's advanced image generation model with exceptional understanding of complex prompts. Excels at photorealistic imagery, accurate object rendering, and following detailed instructions.",
    ),
    "runway-gen4-image": ModelInfo(
        family="runway",
        path="gen4_image",
        external=True,
        references=True,
        description="Runway's Gen-4 image model delivering high-fidelity results with strong coherence. Particularly good at combinging multiple references into a single, cohesive image.",
    ),
    "flux-kontext-1-pro": ModelInfo(
        family="flux_kontext",
        path="black-forest-labs/flux-kontext-pro",
        external=True,
        description="Professional version of FLUX Kontext accessed via API. Offers superior contextual awareness for creating coherent image sequences and variations with consistent themes.",
    ),
    "flux-1-1-pro": ModelInfo(
        family="flux",
        path="black-forest-labs/flux-1.1-pro",
        inpainting_path="black-forest-labs/flux-fill-pro",
        external=True,
        description="Pro version of FLUX 1.1 with enhanced capabilities. Excellent text rendering, sharp details, and consistent style. Ideal for professional illustrations and design work.",
    ),
    "topazlabs-upscale": ModelInfo(
        family="topazlabs",
        path="topazlabs/image-upscale",
        external=True,
        description="Topaz Labs' advanced image upscaling model. Specializes in enhancing image resolution while preserving fine details and textures, ideal for professional photography and print work.",
    ),
}


def generate_model_docs():
    docs = """ # Generate images using various diffusion models.
- External models are processed through their respective APIs.
- Auto Divisor ensures image dimensions are divisible by the specified value.
- ControlNets and IP-Adapters are unified as **references**:
- `style`, `face`, `depth`, `canny`, `pose`, etc.
- These guide the generation but do not change the base mode.
- Guidance Scale controls how strongly the prompt influences the output. Lower values often produce more realism, higher values produce more stylization.
- Image generation mode is chosen automatically:
1. If `mask` is provided → routed to **inpainting** (if supported).
2. Else if `image` is provided → **image-to-image**.
3. If neither `image` nor `mask` is provided → **text-to-image**.
- ⚠ Some models always require an input image:
- Super-resolution models (e.g. ESRGAN, Topaz)
- Enhancement models (upscalers, denoisers)
- Depth/segmentation extractors
Requests without an image for these models will raise an error.
"""

    # List all models alphabetically without family grouping
    docs += "# Available Models\n\n"

    for model_name, model_info in MODEL_CONFIG.items():
        docs += model_info.to_doc_format(model_name)

    return docs


# External models are non-local models processed by external services
TaskName: TypeAlias = Literal["process_image", "process_image_external"]


class References(BaseModel):
    mode: Literal["style", "style-plus", "face", "depth", "canny", "pose"]
    strength: float = 0.5
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
        default="worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes",
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
    references: list[References] = []

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_CONFIG[self.model].family

    @property
    def model_path(self) -> str:
        return MODEL_CONFIG[self.model].path

    @property
    def model_path_inpainting(self) -> str:
        """Get the inpainting model path if available, otherwise normal."""
        inpainting_path = MODEL_CONFIG[self.model].inpainting_path
        if inpainting_path:
            return inpainting_path
        return self.model_path

    @property
    def model_path_image_to_image(self) -> str:
        """Get the image-to-image model path if available, otherwise normal."""
        image_to_image_path = MODEL_CONFIG[self.model].image_to_image_path
        if image_to_image_path:
            return image_to_image_path
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
