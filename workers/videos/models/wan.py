from functools import lru_cache

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from transformers import AutoTokenizer, UMT5EncoderModel

from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    time_info_decorator,
)
from utils.utils import ensure_divisible, get_16_9_resolution, resize_image
from videos.context import VideoContext

# Wan gives better results with a default negative prompt
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


@time_info_decorator
@lru_cache(maxsize=1)
def get_pipeline_text_encoder(torch_dtype=torch.float16):
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    ).to("cpu")

    class TextEncoderWrapper:
        def __init__(self, model: UMT5EncoderModel, tokenizer: AutoTokenizer):
            self.model = model
            self.tokenizer = tokenizer

        @lru_cache(maxsize=5)
        @time_info_decorator
        @torch.no_grad()
        def encode(self, prompt, max_sequence_length=256):
            device = self.model.device
            dtype = self.model.dtype

            text_inputs = self.tokenizer(  # type: ignore
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
            seq_lens = mask.gt(0).sum(dim=1).long()

            prompt_embeds = self.model(text_input_ids.to(device), mask.to(device)).last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
            )

            # ALTERNATIVE: (slightly different behavior for padding tokens)
            # input_ids = text_inputs.input_ids.to(device)
            # attention_mask = text_inputs.attention_mask.to(device)

            # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # prompt_embeds = outputs.last_hidden_state.to(dtype=dtype, device=device)

            # # zero out embeddings for padded tokens to match pipeline behavior
            # prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(dtype)

            return prompt_embeds

    return TextEncoderWrapper(text_encoder, tokenizer)


@decorator_global_pipeline_cache
def get_pipeline_i2v(model_id, wan_2_1=False, torch_dtype=torch.bfloat16) -> WanImageToVideoPipeline:
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    transformer_2 = None
    if wan_2_1:
        transformer_2 = get_quantized_model(
            model_id=model_id,
            subfolder="transformer_2",
            model_class=WanTransformer3DModel,
            target_precision=4 if LOW_VRAM else 4,
            torch_dtype=torch_dtype,
        )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=None,
        tokenizer=None,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
        # boundary_ratio=0.0,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if LOW_VRAM:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


@decorator_global_pipeline_cache
def get_pipeline_t2v(model_id, wan_2_1=False, torch_dtype=torch.bfloat16) -> WanPipeline:
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    transformer_2 = None
    if wan_2_1:
        transformer_2 = get_quantized_model(
            model_id=model_id,
            subfolder="transformer_2",
            model_class=WanTransformer3DModel,
            target_precision=4 if LOW_VRAM else 4,
            torch_dtype=torch_dtype,
        )

    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=None,
        tokenizer=None,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if LOW_VRAM:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def text_to_video(context: VideoContext):
    text_encoder_pipe = get_pipeline_text_encoder()
    prompt_embeds = text_encoder_pipe.encode(context.data.prompt)
    negative_prompt_embeds = text_encoder_pipe.encode(negative_prompt)

    if context.data.model == "wan-2-1":
        pipe = get_pipeline_t2v(model_id="magespace/Wan2.1-T2V-14B-Lightning-Diffusers", wan_2_1=False)
    else:
        pipe = get_pipeline_t2v(model_id="magespace/Wan2.2-T2V-A14B-Lightning-Diffusers", wan_2_1=True)

    width, height = get_16_9_resolution("480p")
    width = ensure_divisible(width, 16)
    height = ensure_divisible(height, 16)

    output = pipe(
        width=width,
        height=height,
        prompt_embeds=prompt_embeds.to("cuda"),
        negative_prompt_embeds=negative_prompt_embeds.to("cuda"),
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
    ).frames[0]

    # NOTE maybe manually clear cuda cache here
    # del prompt_embeds, negative_prompt_embeds

    processed_path = context.save_video(output, fps=16)
    return processed_path


def main(context: VideoContext):
    image = context.image
    if image is None:
        return text_to_video(context)

    text_encoder_pipe = get_pipeline_text_encoder()
    prompt_embeds = text_encoder_pipe.encode(context.data.prompt)
    negative_prompt_embeds = text_encoder_pipe.encode(negative_prompt)

    if context.data.model == "wan-2-1":
        pipe = get_pipeline_i2v(model_id="magespace/Wan2.1-I2V-14B-480P-Lightning-Diffusers", wan_2_1=False)
    else:
        pipe = get_pipeline_i2v(model_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers", wan_2_1=True)

    width, height = get_16_9_resolution("720p")
    image = resize_image(image, 16, 1.0, width, height)

    output = pipe(
        width=image.size[0],
        height=image.size[1],
        image=image,
        prompt_embeds=prompt_embeds.to("cuda"),
        negative_prompt_embeds=negative_prompt_embeds.to("cuda"),
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path
