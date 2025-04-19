from fastapi import APIRouter, HTTPException

from image.context import ImageContext
from image.models.auto_diffusion import main as auto_diffusion
from image.models.depth_anything import main as depth_anything
from image.models.segment_anything import main as segment_anything
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
    elif request.model == "depth-anything" or request.model == "depth_anything":
        main = depth_anything
        mode = "depth"
    elif (
        request.model == "segment-anything" or request.model == "segment_anything" or request.model == "facebook/sam2"
    ):
        main = segment_anything
        mode = "mask"
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
