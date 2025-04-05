from functools import lru_cache

import torch
from diffusers import ControlNetModel, SD3ControlNetModel
from image.schemas import ControlNetSchema
from utils.logger import logger
from utils.utils import cache_info_decorator, load_image_if_exists


@cache_info_decorator
@lru_cache(maxsize=2)
def load_controlnet(model, torch_dtype=torch.float16):
    result = ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
    return result


@cache_info_decorator
@lru_cache(maxsize=2)
def load_sd3_controlnet(model, torch_dtype=torch.float16):
    result = SD3ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
    return result


class ControlNet:
    def __init__(self, data: ControlNetSchema, width, height, torch_dtype=torch.float16):
        self.model = data.model
        self.image_path = data.image_path
        self.conditioning_scale = data.conditioning_scale
        self.enabled = False
        self.enabled = self.model is not None and self.image_path is not None
        self.loaded_controlnet = None
        self.control_net_type = "default"
        if "sd3" in self.model.lower() or "sd-3" in self.model.lower():
            self.control_net_type = "sd3"

        self.image = load_image_if_exists(self.image_path)
        if self.image:
            self.image = self.image.resize([width, height])
        else:
            self.enabled = False

        if self.enabled:
            if self.control_net_type == "sd3":
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
            print("Cleaning up controlnet")
            # free up the memory
            self.loaded_controlnet.to("cpu")
