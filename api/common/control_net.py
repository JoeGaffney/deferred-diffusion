from functools import lru_cache

import torch
from diffusers import ControlNetModel, SD3ControlNetModel

from common.pipeline_helpers import is_model_sd3
from image.schemas import ControlNetSchema
from utils.logger import logger
from utils.utils import cache_info_decorator, load_image_if_exists


@cache_info_decorator
@lru_cache(maxsize=2)
def load_controlnet(model, torch_dtype=torch.float16):
    result = ControlNetModel.from_pretrained(model, variant="fp16", torch_dtype=torch_dtype, device_map="cpu")
    return result


@cache_info_decorator
@lru_cache(maxsize=2)
def load_sd3_controlnet(model, torch_dtype=torch.float16):
    result = SD3ControlNetModel.from_pretrained(model, variant="fp16", torch_dtype=torch_dtype, device_map="cpu")
    return result


class ControlNet:
    def __init__(self, data: ControlNetSchema, width, height, torch_dtype=torch.float16):
        self.model = data.model
        self.image_path = data.image_path
        self.conditioning_scale = data.conditioning_scale
        self.enabled = False
        self.enabled = self.model is not None and self.image_path is not None
        self.loaded_controlnet = None
        self.model_sd3 = is_model_sd3(self.model)

        self.image = load_image_if_exists(self.image_path)
        if self.image:
            self.image = self.image.resize([width, height])
        else:
            self.enabled = False

        if self.conditioning_scale < 0.01:
            self.enabled = False

        if self.enabled:
            if self.model_sd3:
                self.loaded_controlnet = load_sd3_controlnet(self.model, torch_dtype=torch_dtype)
            else:
                self.loaded_controlnet = load_controlnet(self.model, torch_dtype=torch_dtype)

    # we load then offload to match the same behavior as cpu offloading
    def get_loaded_controlnet(self):
        if self.loaded_controlnet is None:
            return None
        return self.loaded_controlnet.to("cuda")

    def cleanup(self):
        if self.loaded_controlnet:
            self.loaded_controlnet.to("cpu")
