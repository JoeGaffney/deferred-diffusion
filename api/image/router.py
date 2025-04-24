import copy

from fastapi import APIRouter, HTTPException

from image.context import ImageContext
from image.models.auto_diffusion import main as auto_diffusion
from image.models.auto_openai import main as auto_openai
from image.models.depth_anything import main as depth_anything
from image.models.segment_anything import main as segment_anything
from image.models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from image.schemas import ImageRequest, ImageResponse

router = APIRouter(prefix="/image", tags=["Image"])


@router.post("", response_model=ImageResponse, operation_id="create_image")
def create(request: ImageRequest):
    context = ImageContext(request)
    mode = context.model_config.mode
    family = context.model_config.model_family

    if mode == "upscale":
        result = stable_diffusion_upscaler(context, mode=mode)
    elif mode == "depth":
        result = depth_anything(context, mode=mode)
    elif mode == "mask":
        result = segment_anything(context, mode=mode)
    else:
        # auto_diffusion
        auto_mode = "img_to_img"
        if context.data.input_mask_path != "":
            auto_mode = "img_to_img_inpainting"
        if context.data.input_image_path == "":
            auto_mode = "text_to_image"

        if family == "openai":
            result = auto_openai(context, mode=auto_mode)
        else:
            result = auto_diffusion(context, mode=auto_mode)

    return ImageResponse(data=result)
