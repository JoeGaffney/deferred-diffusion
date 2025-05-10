from images.context import ImageContext
from images.models.auto_diffusion import main as auto_diffusion
from images.models.auto_openai import main as auto_openai
from images.models.depth_anything import main as depth_anything
from images.models.segment_anything import main as segment_anything
from images.models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from images.schemas import ImageRequest, ImageResponse
from PIL import Image
from utils.utils import pil_to_base64
from worker import celery_app  # Import from worker.py


@celery_app.task(name="process_image")
def process_image(request_dict):

    # Convert dictionary back to proper object
    request = ImageRequest.model_validate(request_dict)
    # raise ValueError("testing if raise is passed through")

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
        if context.data.mask:
            auto_mode = "img_to_img_inpainting"
        if context.data.image is None:
            auto_mode = "text_to_image"

        if family == "openai":
            result = auto_openai(context, mode=auto_mode)
        else:
            result = auto_diffusion(context, mode=auto_mode)

    if isinstance(result, Image.Image):
        # save a temp file for now
        context.save_image(result)
        return ImageResponse(base64_data=pil_to_base64(result)).model_dump()

    raise ValueError("Image generation failed")
