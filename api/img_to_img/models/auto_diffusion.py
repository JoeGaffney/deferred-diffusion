import os
import torch
from functools import lru_cache
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    StableDiffusion3ControlNetPipeline,
)
from utils.diffusers_helpers import (
    diffusers_image_call,
    optimize_pipeline,
    diffusers_call,
    diffusers_inpainting_call,
)
from utils.utils import get_16_9_resolution
from common.context import Context


@lru_cache(maxsize=4)  # Cache up to 4 different pipelines
def get_pipeline(model_id, torch_dtype=torch.float16, disable_text_encoder_3=True):

    # this can really eat up the memory
    pipe = None
    if disable_text_encoder_3 == True:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            text_encoder_3=None,
            tokenizer_3=None,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    print("loaded pipeline", model_id, torch_dtype)

    return optimize_pipeline(pipe)


def get_text_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    if controlnets != []:
        return AutoPipelineForText2Image.from_pipe(
            get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
            requires_safety_checker=False,
            controlnet=controlnets,
        )

    return AutoPipelineForText2Image.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
    )


def get_image_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    if controlnets != []:
        return AutoPipelineForImage2Image.from_pipe(
            get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
            requires_safety_checker=False,
            controlnet=controlnets,
        )

    return AutoPipelineForImage2Image.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
    )


def get_inpainting_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    if controlnets != []:
        return AutoPipelineForInpainting.from_pipe(
            get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
            requires_safety_checker=False,
            controlnet=controlnets,
        )

    return AutoPipelineForInpainting.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
    )


# need to grab direct as SD3 control nets not full supported by diffusers
def get_sd3_controlnet_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    pipe = None
    pipe = StableDiffusion3ControlNetPipeline.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        controlnet=controlnets,
    )

    print("loaded pipeline", model_id, torch_dtype, controlnets)
    return optimize_pipeline(pipe)


# work around as SD3 control nets not full supported by diffusers
def main_sd_3_controlnets(context: Context, model_id="stabilityai/stable-diffusion-3-medium-diffusers", mode="text"):
    disable_text_encoder_3 = context.disable_text_encoder_3
    controlnets = context.get_loaded_controlnets()
    torch_dtype = context.torch_dtype

    # work around as SD3 not full supported by diffusers
    if mode == "text_to_image":
        return diffusers_call(
            get_sd3_controlnet_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
            use_image_wh=False,
        )
    elif mode == "img_to_img":
        # there is no dedicated img to img control net for SD3
        return diffusers_call(
            get_sd3_controlnet_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
            use_image_wh=True,
        )

    elif mode == "img_to_img_inpainting":
        # there is no dedicated img to img inpainting control net for SD3
        return diffusers_call(
            get_sd3_controlnet_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
            use_image_wh=True,
        )

    return "invalid mode"


def main(
    context: Context,
    model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    mode="text",
):
    if context.sd3_controlnet_mode == True:
        return main_sd_3_controlnets(context, model_id=model_id, mode=mode)

    disable_text_encoder_3 = context.disable_text_encoder_3
    controlnets = context.get_loaded_controlnets()
    torch_dtype = context.torch_dtype

    if mode == "text_to_image":
        return diffusers_call(
            get_text_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )
    elif mode == "img_to_img":
        return diffusers_image_call(
            get_image_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )
    elif mode == "img_to_img_inpainting":
        return diffusers_inpainting_call(
            get_inpainting_pipeline(
                model_id,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )

    return "invalid mode"


def validation_tests(
    output_name,
    model_ids=[
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
    ],
    controlnets=[],
    prompt="Detailed, 8k, DSLR photo, photorealistic, Eye, enchance keep original elements",
):
    width, height = get_16_9_resolution("540p")

    prompt = "Detailed, 8k, DSLR photo, photorealistic, Eye, enchance keep original elements"

    for model_id in model_ids:
        for mode in ["text_to_image", "img_to_img", "img_to_img_inpainting"]:
            model_id_nice = model_id.replace("/", "_")

            main(
                Context(
                    model=model_id,
                    input_image_path="../tmp/tornado_v001.JPG",
                    input_mask_path="../tmp/tornado_v001_mask.png",
                    output_image_path=f"../tmp/output/{model_id_nice}/{output_name}_{mode}.png",
                    prompt=prompt,
                    strength=0.5,
                    guidance_scale=7.5,
                    max_width=width,
                    max_height=height,
                    controlnets=controlnets,
                ),
                model_id=model_id,
                mode=mode,
            )


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    controlnet_a = {
        "model": "InstantX/SD3-Controlnet-Canny",
        "input_image": "../tmp/canny.png",
        "conditioning_scale": "0.5",
    }
    controlnet_b = {
        "model": "diffusers/controlnet-canny-sdxl-1.0",
        "input_image": "../tmp/canny.png",
        "conditioning_scale": "0.5",
    }

    validation_tests(
        output_name,
        model_ids=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
        ],
        controlnets=[],
        prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enchance keep original elements",
    )

    validation_tests(
        output_name + "_controlnets",
        model_ids=[
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
        controlnets=[controlnet_b, controlnet_b],
    )

    validation_tests(
        output_name + "_controlnets",
        model_ids=[
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
        ],
        controlnets=[controlnet_a, controlnet_a],
    )
