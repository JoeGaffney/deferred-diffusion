import os
from functools import lru_cache

import torch
from diffusers import ControlNetModel, SD3ControlNetModel
from utils.logger import logger


@lru_cache(maxsize=4)  # Cache up to 4 different controlnets
def load_controlnet(model, torch_dtype=torch.float16):
    try:
        controlnet = ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
        logger.warn("loaded controlnet", model, torch_dtype)
    except:
        logger.warn("Failed to load ControlNetModel", model, torch_dtype)

    try:
        controlnet = SD3ControlNetModel.from_pretrained(model, torch_dtype=torch_dtype)
        logger.warn("loaded controlnet", model, torch_dtype)
    except:
        logger.warn("Failed to load SD3ControlNetModel", model, torch_dtype)

    # if not setting cuda here we get clashes with the main model unsure if the controlnet inherits the cup offload
    controlnet.to("cuda")
    return controlnet


class ControlNet:
    def __init__(self, data, torch_dtype=torch.float16):
        self.model = data.get("model")
        self.input_image = data.get("input_image")
        self.conditioning_scale = float(data.get("conditioning_scale", 0.5))
        self.enabled = False
        self.is_valid = self.model is not None and self.input_image is not None
        self.loaded_controlnet = None

        # validate input image
        if self.is_valid:
            if not os.path.exists(self.input_image):
                self.is_valid = False

        if self.is_valid:
            self.loaded_controlnet = load_controlnet(self.model, torch_dtype=torch_dtype)

        if self.loaded_controlnet is None:
            self.is_valid = False
