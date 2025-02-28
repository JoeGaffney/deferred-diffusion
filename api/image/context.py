import copy
import math
import os
from typing import List

import requests
import torch
from common.control_net import ControlNet
from diffusers.utils import export_to_video, load_image
from image.schemas import ImageRequest
from utils import device_info
from utils.logger import logger
from utils.utils import ensure_path_exists, save_copy_with_timestamp


def is_model_sd3(model):
    return "stable-diffusion-3" in model


class ImageContext:
    def __init__(self, data: ImageRequest):
        self.data = data
        self.model = data.model
        self.orig_height = copy.copy(data.max_height)
        self.orig_width = copy.copy(data.max_width)
        self.torch_dtype = torch.float16  # just keep float 16 for now

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
            tmp = ControlNet(current, torch_dtype=self.torch_dtype)
            if tmp.is_valid:
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

    def resize_image(self, image, division=16, scale=1.0):

        # Ensure the new dimensions do not exceed max_width and max_height
        width = min(image.size[0] * scale, self.data.max_width)
        height = min(image.size[1] * scale, self.data.max_height)

        # Adjust width and height to be divisible by 32 or 8
        width = math.ceil(width / division) * division
        height = math.ceil(height / division) * division

        logger.info(f"Image Resized from: {image.size} to {width}x{height}")
        return image.resize((width, height))

    def resize_max_wh(self, division=16):
        width = math.ceil(self.data.max_width / division) * division
        height = math.ceil(self.data.max_height / division) * division
        return width, height

    def resize_image_to_orig(self, image, scale=1):
        return image.resize((self.orig_width * scale, self.orig_height * scale))

    def resize_image_to_max_wh(self, image, scale=1):
        return image.resize((self.data.max_width * scale, self.data.max_height * scale))

    def load_image(self, division=8, scale=1.0):
        if not os.path.exists(self.data.input_image_path):
            raise FileNotFoundError(self.data.input_image_path)

        image = load_image(self.data.input_image_path)

        logger.info(f"Image loaded from {self.data.input_image_path} size: {image.size}")
        self.orig_width, self.orig_height = image.size

        tmp = self.resize_image(image, division, scale)
        return tmp

    def load_mask(self, size):
        if not os.path.exists(self.data.input_mask_path):
            raise FileNotFoundError(self.data.input_mask_path)

        image = load_image(self.data.input_mask_path)
        image = image.convert("L")

        # ensure he same size as the color image
        image = image.resize(size)
        logger.info(f"Mask loaded from {self.data.input_mask_path} size: {image.size}")
        return image

    def get_controlnet_images(self, size):
        if self.controlnets_enabled == False:
            return []

        images = []
        for controlnet in self.controlnets:
            image = load_image(controlnet.input_image)

            # ensure the same size as the color image
            image = image.resize(size)
            images.append(image)
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
