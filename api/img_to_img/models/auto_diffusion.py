import os
import torch
from functools import lru_cache
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
    DiffusionPipeline,
)
from utils.diffusers_helpers import (
    diffusers_image_call,
    optimize_pipeline,
    diffusers_call,
    diffusers_inpainting_call,
)
from utils.utils import get_16_9_resolution
from common.context import Context


torch_dtype_map = {
    "stabilityai/stable-diffusion-3-medium-diffusers": torch.float16,
    "stabilityai/stable-diffusion-3.5-medium": torch.bfloat16,
    "stabilityai/stable-diffusion-xl-base-1.0": torch.float16,
    "stabilityai/stable-diffusion-xl-refiner-1.0": torch.float16,
}


@lru_cache(maxsize=4)  # Cache up to 4 different pipelines
def get_pipeline(model_id):
    torch_dtype = torch_dtype_map.get(model_id, torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    print("loaded pipeline", model_id, torch_dtype, model_id)
    return optimize_pipeline(pipe)


def get_text_pipeline(model_id, control_nets=[]):
    return AutoPipelineForText2Image.from_pipe(
        get_pipeline(model_id), control_nets=control_nets, requires_safety_checker=False
    )


def get_image_pipeline(model_id, control_nets=[]):
    return AutoPipelineForImage2Image.from_pipe(
        get_pipeline(model_id), control_nets=control_nets, requires_safety_checker=False
    )


def get_inpainting_pipeline(model_id, control_nets=[]):
    return AutoPipelineForInpainting.from_pipe(
        get_pipeline(model_id), control_nets=control_nets, requires_safety_checker=False
    )


def main(
    context: Context,
    model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    mode="text",
):
    if mode == "text_to_image":
        return diffusers_call(get_text_pipeline(model_id), context)
    elif mode == "img_to_img":
        return diffusers_image_call(get_image_pipeline(model_id), context)
    elif mode == "img_to_img_inpainting":
        return diffusers_inpainting_call(get_inpainting_pipeline(model_id), context)

    return "invalid mode"


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]
    width, height = get_16_9_resolution("540p")

    for model_id in [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
    ]:
        for mode in ["text_to_image", "img_to_img", "img_to_img_inpainting"]:
            model_id_nice = model_id.replace("/", "_")

            main(
                Context(
                    input_image_path="../tmp/tornado_v001.JPG",
                    input_mask_path="../tmp/tornado_v001_mask.png",
                    output_image_path=f"../tmp/output/{model_id_nice}/{output_name}_{mode}.png",
                    prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enchance keep original elements",
                    strength=0.5,
                    guidance_scale=7.5,
                    max_width=width,
                    max_height=height,
                ),
                model_id=model_id,
                mode=mode,
            )
