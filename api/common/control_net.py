from functools import lru_cache

import torch
from diffusers import ControlNetModel, SD3ControlNetModel

from common.exeptions import ControlNetConfigError
from image.schemas import ControlNetSchema, ModelConfig
from utils.utils import cache_info_decorator, load_image_if_exists


@cache_info_decorator
@lru_cache(maxsize=2)
def load_controlnet(model, torch_dtype=torch.float16):
    result = ControlNetModel.from_pretrained(model, variant="fp16", torch_dtype=torch_dtype, device_map="cpu")
    return result


@cache_info_decorator
@lru_cache(maxsize=2)
def load_sd3_controlnet(model, torch_dtype=torch.float16):
    result = SD3ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype, device_map="cpu")
    return result


CONTROL_NET_MODEL_CONFIG = {
    "sd1.5": {
        "depth": "lllyasviel/sd-controlnet-depth",
        "canny": "lllyasviel/sd-controlnet-canny",
        "pose": "lllyasviel/sd-controlnet-openpose",
    },
    "sdxl": {
        "depth": "diffusers/controlnet-depth-sdxl-1.0-small",
        "canny": "diffusers/controlnet-canny-sdxl-1.0-small",
        "pose": "xinsir/controlnet-openpose-sdxl-1.0",
    },
    "sd3": {
        "depth": "InstantX/SD3-Controlnet-Depth",
        "canny": "InstantX/SD3-Controlnet-Canny",
    },
    "flux": {
        "depth": "XLabs-AI/flux-controlnet-depth-v3",
        "canny": "XLabs-AI/flux-controlnet-canny-v3",
    },
}


def get_controlnet_model(model_family: str, controlnet_model: str) -> str:
    model_config = CONTROL_NET_MODEL_CONFIG.get(model_family)
    if not model_config:
        raise ControlNetConfigError(f"ControlNet model config for {model_family} not found")

    controlnet_model_path = model_config.get(controlnet_model)
    if not controlnet_model_path:
        raise ControlNetConfigError(f"ControlNet model path for {controlnet_model} not found in {model_family}")

    return controlnet_model_path


class ControlNet:
    def __init__(self, data: ControlNetSchema, model_config: ModelConfig, width, height, torch_dtype=torch.float16):
        self.model = get_controlnet_model(model_config.model_family, data.model)
        self.image_path = data.image_path
        self.conditioning_scale = data.conditioning_scale
        self.image = load_image_if_exists(self.image_path)

        if self.conditioning_scale < 0.01:
            raise ControlNetConfigError("ControlNet conditioning scale must be >= 0.01")

        if not self.image:
            raise ControlNetConfigError(f"Could not load ControlNet image from {self.image_path}")

        self.image = self.image.resize([width, height])

        if model_config.model_family == "sd3":
            self.loaded_controlnet = load_sd3_controlnet(self.model, torch_dtype=torch_dtype)

        self.loaded_controlnet = load_controlnet(self.model, torch_dtype=torch_dtype)

    # we load then offload to match the same behavior as cpu offloading
    def get_loaded_controlnet(self):
        return self.loaded_controlnet.to("cuda")

    def cleanup(self):
        self.loaded_controlnet.to("cpu")
