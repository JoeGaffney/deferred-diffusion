import warnings

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from transformers import CLIPVisionModel, UMT5EncoderModel

from common.logger import logger
from common.memory import LOW_VRAM
from common.pipeline_helpers import get_quantized_model
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext

WAN_TRANSFORMER_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
UMT_T5_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


class IgnoreRotaryWarning:
    def __enter__(self):
        self._orig_warn = warnings.warn

        def custom_warn(message, *args, **kwargs):
            if "cross_attention_kwargs ['rotary_emb'] are not expected" not in str(message):
                self._orig_warn(message, *args, **kwargs)

        warnings.warn = custom_warn

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.warn = self._orig_warn


# NOTE this one is heavy maybe we should not cache it globally
# @decorator_global_pipeline_cache
def get_pipeline(model_id, torch_dtype=torch.bfloat16):
    image_encoder = get_quantized_model(
        model_id=model_id,
        subfolder="image_encoder",
        model_class=CLIPVisionModel,
        target_precision=16,
        torch_dtype=torch.float32,
    )

    transformer = get_quantized_model(
        model_id=WAN_TRANSFORMER_MODEL_PATH,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=8,
        torch_dtype=torch_dtype,
    )

    text_encoder = get_quantized_model(
        model_id=UMT_T5_MODEL_PATH,
        subfolder="text_encoder",
        model_class=UMT5EncoderModel,
        target_precision=8,
        torch_dtype=torch_dtype,
    )

    vae = get_quantized_model(
        model_id=model_id,
        subfolder="vae",
        model_class=AutoencoderKLWan,
        target_precision=16,
        torch_dtype=torch.float32,
    )

    # Alt schedulers
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    # scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        image_encoder=image_encoder,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    # pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if LOW_VRAM:
        logger.warning("Enabling sequential CPU offload for low VRAM mode.")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    return pipe


def main(context: VideoContext):
    pipe = get_pipeline(model_id=context.data.model_path)
    image = context.image
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")

    width, height = get_16_9_resolution("720p")
    image = resize_image(image, 16, 1.0, width, height)

    with IgnoreRotaryWarning():
        output = pipe(
            width=image.size[0],
            height=image.size[1],
            image=image,
            prompt=context.data.prompt,
            negative_prompt=context.data.negative_prompt,
            num_inference_steps=context.data.num_inference_steps,
            num_frames=context.data.num_frames,
            guidance_scale=context.data.guidance_scale,
            generator=context.get_generator(),
        ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path
