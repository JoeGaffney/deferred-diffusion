from functools import lru_cache

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    HiDreamImagePipeline,
    HiDreamImageTransformer2DModel,
    StableDiffusionXLPipeline,
)
from PIL import Image
from transformers import (
    CLIPVisionModelWithProjection,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5EncoderModel,
)

from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from images.context import ImageContext, PipelineConfig

# this is common and shared accross a few models use the flux one for now it should be google/t5-v1_1-xxl
T5_MODEL_PATH = "google/t5-efficient-mini"  # use the original one for now
T5_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"
LLAMA_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def get_pipeline_flux(config: PipelineConfig):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=config.model_id,
        subfolder="transformer",
        model_class=FluxTransformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder_2"] = get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )
    if config.ip_adapter_models != ():
        if not hasattr(pipe, "load_ip_adapter"):
            raise ValueError("The pipeline does not support IP-Adapters. Please use a compatible pipeline.")

        # load multiple as arrays
        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            weight_name=list(config.ip_adapter_weights),
            image_encoder_pretrained_model_name_or_path=config.ip_adapter_image_encoder_subfolder,
        )

    return optimize_pipeline(pipe, sequential_cpu_offload=False)


def get_pipeline_high_dream(config: PipelineConfig):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=config.model_id,
        subfolder="transformer",
        model_class=HiDreamImageTransformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    args["text_encoder_3"] = get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    args["text_encoder_4"] = get_quantized_model(
        model_id=LLAMA_MODEL_PATH,
        subfolder="",
        model_class=LlamaForCausalLM,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )
    # NOTE ref from the model card
    args["text_encoder_4"].output_hidden_states = True
    args["text_encoder_4"].output_attentions = True

    args["tokenizer_4"] = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH)

    pipe = HiDreamImagePipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )

    return optimize_pipeline(pipe, sequential_cpu_offload=False)


@decorator_global_pipeline_cache
def get_pipeline(config: PipelineConfig):
    if config.model_family == "flux":
        return get_pipeline_flux(config)
    elif config.model_family == "hidream":
        return get_pipeline_high_dream(config)

    args = {"torch_dtype": config.torch_dtype, "use_safetensors": True}

    # this can really eat up the memory
    if config.model_family == "sd3":
        args["text_encoder_3"] = None
        args["tokenizer_3"] = None

    if config.model_id == "RunDiffusion/Juggernaut-XL-v9":
        # NOTE see https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/discussions/6
        pipe = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            **args,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            config.model_id,
            **args,
        )

    if config.ip_adapter_models != ():
        if not hasattr(pipe, "load_ip_adapter"):
            raise ValueError("The pipeline does not support IP-Adapters. Please use a compatible pipeline.")

        # load multiple as arrays
        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            subfolder=list(config.ip_adapter_subfolders),
            weight_name=list(config.ip_adapter_weights),
        )

        # image_encoder is require for some adapters
        if config.ip_adapter_image_encoder_model != "":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.ip_adapter_image_encoder_model,
                subfolder=config.ip_adapter_image_encoder_subfolder,
                torch_dtype=config.torch_dtype,
            )
            pipe.image_encoder = image_encoder

            # Supposed to help with consistency
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return optimize_pipeline(pipe, sequential_cpu_offload=False)


def setup_controlnets_and_ip_adapters(pipe, context: ImageContext, args):
    if context.control_nets.is_enabled():
        if context.data.model_family == "sd3" or context.data.model_family == "flux":
            args["control_image"] = context.control_nets.get_images()
        else:
            args["image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        if context.data.model_family != "flux":
            args["cross_attention_kwargs"] = {"ip_adapter_masks": context.adapters.get_masks()}
        pipe = context.adapters.set_scale(pipe)

    # NOTE there is a bug when using controlnets and ip adapters together with flux
    # if context.data.model_family == "flux":
    #     num_adapters = pipe.transformer.encoder_hid_proj.num_ip_adapters
    #     adapter_images = args.get("ip_adapter_image", [])
    #     logger.info(f"Flux model has {num_adapters} IP adapters and {len(adapter_images)} images")

    return pipe, args


def text_to_image_call(context: ImageContext):

    def get_text_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        if pipeline_config.model_family == "hidream":
            return get_pipeline(pipeline_config)

        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForText2Image.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_text_pipeline(context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    processed_image = context.resize_image_to_orig(processed_image)
    return processed_image


def image_to_image_call(context: ImageContext):

    def get_image_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        if pipeline_config.model_family == "hidream":
            return get_pipeline(pipeline_config)

        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForImage2Image.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_image_pipeline(context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": context.color_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
    }

    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    processed_image = context.resize_image_to_orig(processed_image)
    return processed_image


def inpainting_call(context: ImageContext):

    def get_inpainting_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        if pipeline_config.model_family == "hidream":
            return get_pipeline(pipeline_config)

        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForInpainting.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_inpainting_pipeline(
        context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets()
    )
    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
        # "padding_mask_crop": 32,
    }
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Inpainting call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    processed_image = context.resize_image_to_orig(processed_image)
    return processed_image


def main(context: ImageContext) -> Image.Image:
    pipeline_config = context.get_pipeline_config()

    mode = "img_to_img"
    if context.data.mask:
        mode = "img_to_img_inpainting"
    if context.data.image is None:
        mode = "text_to_image"

    # work around as SD3 not fully supported by diffusers
    if context.control_nets.is_enabled() and pipeline_config.model_family == "sd3":
        return text_to_image_call(context)

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")
