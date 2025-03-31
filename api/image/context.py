import copy
from typing import List

import torch
from common.control_net import ControlNet
from image.schemas import ImageRequest
from utils import device_info
from utils.logger import logger
from utils.utils import (
    ensure_path_exists,
    load_image_if_exists,
    resize_image,
    save_copy_with_timestamp,
)


def is_model_sd3(model):
    return "stable-diffusion-3" in model


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.data = data
        self.model = data.model
        self.orig_height = copy.copy(data.max_height)
        self.orig_width = copy.copy(data.max_width)

        # Round down to nearest multiple of 16
        self.division = 16
        self.width = (copy.copy(data.max_width) // self.division) * self.division
        self.height = (copy.copy(data.max_height) // self.division) * self.division

        self.torch_dtype = torch.float16  # just keep float 16 for now

        self.color_image = load_image_if_exists(data.input_image_path)
        if self.color_image:
            self.color_image = resize_image(
                self.color_image, self.division, 1.0, self.data.max_width, self.data.max_height
            )
            self.orig_width, self.orig_height = self.color_image.size
            self.width, self.height = self.color_image.size

        self.mask_image = load_image_if_exists(data.input_mask_path)
        if self.mask_image:
            self.mask_image = self.mask_image.convert("L")
            self.mask_image = self.mask_image.resize([self.width, self.height])

        ensure_path_exists(self.data.output_image_path)
        ensure_path_exists(self.data.input_image_path)

        # SD3 models disabling the text encoder 3 can reduce vram usage
        self.model_sd3 = is_model_sd3(data.model)
        self.disable_text_encoder_3 = True
        if self.model_sd3 and bool(data.disable_text_encoder_3) == False:
            self.disable_text_encoder_3 = False

        # add our control nets
        self.controlnets: List[ControlNet] = []
        for current in data.controlnets:
            tmp = ControlNet(current, self.width, self.height, torch_dtype=self.torch_dtype)
            if tmp.enabled:
                self.controlnets.append(tmp)

        self.controlnets_enabled = len(self.controlnets) > 0

        # requires different path as auto diffusers don't map the control net models for sd3 variants
        self.sd3_controlnet_mode = self.model_sd3 and self.controlnets_enabled

    def save_image(self, image):
        path = self.data.output_image_path
        image.save(path)
        logger.info(f"Image saved at {path} size: {image.size}")

        save_copy_with_timestamp(path)
        return path

    def resize_image_to_orig(self, image, scale=1):
        return image.resize((self.orig_width * scale, self.orig_height * scale))

    def get_controlnet_images(self):
        if self.controlnets_enabled == False:
            return []

        images = []
        for controlnet in self.controlnets:
            images.append(controlnet.image)
        return images

    def get_controlnet_conditioning_scales(self):
        if self.controlnets_enabled == False:
            return 0.8

        controlnet_conditioning_scales = []
        for controlnet in self.controlnets:
            controlnet_conditioning_scales.append(controlnet.conditioning_scale)
        return controlnet_conditioning_scales

    def get_loaded_controlnets(self):
        if self.controlnets_enabled == False:
            return []

        loaded_controlnets = []
        for controlnet in self.controlnets:
            loaded_controlnets.append(controlnet.loaded_controlnet)
        return loaded_controlnets
