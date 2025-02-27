from common.context import Context
from fastapi import APIRouter, HTTPException, Request
from image.models.auto_diffusion import main as auto_diffusion
from image.models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from pydantic import BaseModel

router = APIRouter(prefix="/image", tags=["Image"])


class ImageRequest(BaseModel):
    controlnets: list = []
    disable_text_encoder_3: bool = True
    guidance_scale: float = 10.0
    inpainting_full_image: bool = True
    input_image_path: str = ""
    input_mask_path: str = ""
    max_height: int = 2048
    max_width: int = 2048
    model: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: int = 48
    num_inference_steps: int = 25
    output_image_path: str = ""
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5


class ImageResponse(BaseModel):
    data: str


@router.post("/", response_model=ImageResponse)
def create(request: ImageRequest):
    context = Context(
        input_image_path=request.input_image_path,
        input_mask_path=request.input_mask_path,
        output_image_path=request.output_image_path,
        max_height=request.max_height,
        max_width=request.max_width,
        negative_prompt=request.negative_prompt,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        prompt=request.prompt,
        seed=request.seed,
        strength=request.strength,
        guidance_scale=request.guidance_scale,
        inpainting_full_image=request.inpainting_full_image,
        disable_text_encoder_3=request.disable_text_encoder_3,
        controlnets=request.controlnets,
        model=request.model,
    )

    main = None
    if request.model == "stabilityai/stable-diffusion-x4-upscaler":
        main = stable_diffusion_upscaler
        mode = "upscale"
    else:
        mode = "img_to_img"
        if context.input_mask_path != "":
            mode = "img_to_img_inpainting"
        if request.model == "stabilityai/stable-diffusion-xl-refiner-1.0":
            mode = "img_to_img"
        if context.input_image_path == "":
            mode = "text_to_image"
        main = auto_diffusion

    if not main:
        raise HTTPException(status_code=400, detail="Invalid model")

    result = main(context, model_id=request.model, mode=mode)
    return ImageResponse(data=result)
