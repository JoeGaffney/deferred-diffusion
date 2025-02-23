import copy
import math
import os
from typing import List
from diffusers.utils import export_to_video, load_image
import torch
import requests
from common.control_net import ControlNet
from utils.utils import ensure_path_exists, save_copy_with_timestamp
from utils.logger import logger
from utils import device_info


def is_model_sd3(model):
    return "stable-diffusion-3" in model


class Context:
    def __init__(
        self,
        model="",
        guidance_scale=10.0,
        input_image_path="",
        input_mask_path="",
        max_height=2048,
        max_width=2048,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        num_frames=48,
        num_inference_steps=25,
        output_image_path="../tmp/output/processed.png",
        output_video_path="../tmp/output/processed.mp4",
        prompt="Detailed, 8k, photorealistic",
        seed=42,
        strength=0.5,
        disable_text_encoder_3=True,
        inpainting_full_image=True,
        controlnets=[],
        messages=[],
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.input_image_path = input_image_path
        self.input_mask_path = input_mask_path
        self.max_height = max_height
        self.max_width = max_width
        self.negative_prompt = negative_prompt
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.orig_height = copy.copy(max_height)
        self.orig_width = copy.copy(max_width)
        self.output_image_path = output_image_path
        self.output_video_path = output_video_path
        self.prompt = prompt
        self.seed = seed
        self.strength = strength
        self.inpainting_full_image = bool(inpainting_full_image)
        self.torch_dtype = torch.float16  # just keep float 16 for now
        self.messages = messages

        ensure_path_exists(self.output_image_path)
        ensure_path_exists(self.output_video_path)
        ensure_path_exists(self.input_image_path)

        # SD3 models disabling the text encoder 3 can reduce vram usage
        self.model_sd3 = is_model_sd3(model)
        self.disable_text_encoder_3 = True
        if self.model_sd3 and bool(disable_text_encoder_3) == False:
            self.disable_text_encoder_3 = False

        # add our control nets
        self.controlnets: List[ControlNet] = []
        for current in controlnets:
            tmp = ControlNet(current, torch_dtype=self.torch_dtype)
            if tmp.is_valid:
                self.controlnets.append(tmp)

        self.controlnets_enabled = len(self.controlnets) > 0

        # requires different path as auto diffusers don't map the control net models for sd3 variants
        self.sd3_controlnet_mode = self.model_sd3 and self.controlnets_enabled

    def to_dict(self):
        self.log(f"Context settings: {self.__dict__}")
        return self.__dict__

    def save_image(self, image):
        path = self.output_image_path
        image.save(path)
        self.log(f"Image saved at {path} size: {image.size}")

        save_copy_with_timestamp(path)
        return path

    def save_video(self, video, fps=24):
        path = self.output_video_path
        export_to_video(video, path, fps=fps)
        self.log(f"Video saved at {path}")

        save_copy_with_timestamp(path)
        return path

    def save_video_url(self, url):
        path = self.output_video_path

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            self.log(f"Video saved at {path}")
            save_copy_with_timestamp(path)
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path

    def log(self, message):
        logger.info(message)

    def resize_image(self, image, division=16, scale=1.0):

        # Ensure the new dimensions do not exceed max_width and max_height
        width = min(image.size[0] * scale, self.max_width)
        height = min(image.size[1] * scale, self.max_height)

        # Adjust width and height to be divisible by 32 or 8
        width = math.ceil(width / division) * division
        height = math.ceil(height / division) * division

        self.log(f"Image Resized from: {image.size} to {width}x{height}")
        return image.resize((width, height))

    def resize_max_wh(self, division=16):
        width = math.ceil(self.max_width / division) * division
        height = math.ceil(self.max_height / division) * division
        return width, height

    def resize_image_to_orig(self, image, scale=1):
        return image.resize((self.orig_width * scale, self.orig_height * scale))

    def resize_image_to_max_wh(self, image, scale=1):
        return image.resize((self.max_width * scale, self.max_height * scale))

    def load_image(self, division=8, scale=1.0):
        if not os.path.exists(self.input_image_path):
            raise FileNotFoundError(self.input_image_path)

        image = load_image(self.input_image_path)

        self.log(f"Image loaded from {self.input_image_path} size: {image.size}")
        self.orig_width, self.orig_height = image.size

        tmp = self.resize_image(image, division, scale)
        return tmp

    def load_mask(self, size):
        if not os.path.exists(self.input_mask_path):
            raise FileNotFoundError(self.input_mask_path)

        image = load_image(self.input_mask_path)
        image = image.convert("L")

        # ensure he same size as the color image
        image = image.resize(size)
        self.log(f"Mask loaded from {self.input_mask_path} size: {image.size}")
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
