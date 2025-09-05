import os
from functools import lru_cache

import psutil
import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from transformers import TorchAoConfig, UMT5EncoderModel

from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    time_info_decorator,
)


@lru_cache(maxsize=1)
@time_info_decorator
def get_pipeline_wan_text_encoder(torch_dtype=torch.float32, device="cpu"):
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = WanPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=None,
        transformer_2=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: WanPipeline):
            self.pipe = pipe

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=256):

            device = self.pipe.device
            dtype = self.pipe.dtype

            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=256,
                device=device,
            )
            return prompt_embeds

    return TextEncoderWrapper(pipe)
