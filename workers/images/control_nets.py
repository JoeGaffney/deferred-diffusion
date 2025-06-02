from functools import lru_cache

import torch
from diffusers import ControlNetModel, FluxControlNetModel, SD3ControlNetModel

from common.exceptions import ControlNetConfigError
from common.logger import logger
from images.schemas import ControlNetSchema, ModelConfig
from utils.utils import cache_info_decorator, load_image_if_exists


@cache_info_decorator
@lru_cache(maxsize=2)
def load_controlnet(model, model_family, torch_dtype=torch.float16):

    if model_family == "sd3":
        return SD3ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype, device_map="cpu")
    elif model_family == "flux":
        return FluxControlNetModel.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cpu")

    return ControlNetModel.from_pretrained(model, variant="fp16", torch_dtype=torch_dtype, device_map="cpu")


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
        "depth": "XLabs-AI/flux-controlnet-depth-diffusers",
        "canny": "XLabs-AI/flux-controlnet-canny-diffusers",
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
        self.conditioning_scale = data.conditioning_scale
        self.image = load_image_if_exists(data.image)

        if self.conditioning_scale < 0.01:
            raise ControlNetConfigError("ControlNet conditioning scale must be >= 0.01")

        if not self.image:
            raise ControlNetConfigError(f"Could not load ControlNet image from {data.image}")

        self.image = self.image.resize([width, height])
        self.loaded_controlnet = load_controlnet(self.model, model_config.model_family, torch_dtype=torch_dtype)

    # we load then offload to match the same behavior as cpu offloading
    def get_loaded_controlnet(self):
        return self.loaded_controlnet.to("cuda")

    def cleanup(self):
        self.loaded_controlnet.to("cpu")


class ControlNets:
    def __init__(
        self, control_nets: list[ControlNetSchema], model_config: ModelConfig, width, height, torch_dtype=torch.float16
    ):
        self.control_nets: list[ControlNet] = []

        # Handle initialization errors and create valid control nets
        for data in control_nets:
            try:
                control_net = ControlNet(data, model_config, width, height, torch_dtype)
                self.control_nets.append(control_net)
            except ControlNetConfigError as e:
                logger.error(f"Failed to initialize ControlNet: {e}")

    def is_enabled(self) -> bool:
        """Check if there are any valid control nets."""
        return len(self.control_nets) > 0

    def get_loaded_controlnets(self):
        if not self.is_enabled():
            return []
        return [control_net.get_loaded_controlnet() for control_net in self.control_nets]

    def get_images(self):
        if not self.is_enabled():
            return []
        return [control_net.image for control_net in self.control_nets]

    def get_conditioning_scales(self):
        if not self.is_enabled():
            return 0.8
        return [control_net.conditioning_scale for control_net in self.control_nets]

    def cleanup(self):
        for control_net in self.control_nets:
            control_net.cleanup()
