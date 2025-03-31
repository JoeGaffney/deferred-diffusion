from fastapi import APIRouter, HTTPException, Request
from image.context import ImageContext
from image.models.auto_diffusion import main as auto_diffusion
from image.models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from image.schemas import ImageRequest, ImageResponse

router = APIRouter(prefix="/image", tags=["Image"])


@router.post("", response_model=ImageResponse, operation_id="create_image")
def create(request: ImageRequest):
    context = ImageContext(request)

    main = None
    if request.model == "stabilityai/stable-diffusion-x4-upscaler":
        main = stable_diffusion_upscaler
        mode = "upscale"
    else:
        mode = "img_to_img"
        if context.data.input_mask_path != "":
            mode = "img_to_img_inpainting"
        if request.model == "stabilityai/stable-diffusion-xl-refiner-1.0":
            mode = "img_to_img"
        if context.data.input_image_path == "":
            mode = "text_to_image"
        main = auto_diffusion

    if not main:
        raise HTTPException(status_code=400, detail="Invalid model")

    result = main(context, mode=mode)
    return ImageResponse(data=result)
