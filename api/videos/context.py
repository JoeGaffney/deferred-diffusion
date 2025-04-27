import copy
import math
import os

import requests
import torch
from diffusers.utils import export_to_video, load_image

from common.logger import logger
from utils.utils import ensure_path_exists, save_copy_with_timestamp
from videos.schemas import VideoRequest


class VideoContext:
    def __init__(self, data: VideoRequest):
        self.data = data
        self.model = data.model
        self.orig_height = copy.copy(data.max_height)
        self.orig_width = copy.copy(data.max_width)
        self.torch_dtype = torch.float16  # just keep float 16 for now

        ensure_path_exists(self.data.output_video_path)
        ensure_path_exists(self.data.input_image_path)

    def save_video(self, video, fps=24):
        path = self.data.output_video_path
        export_to_video(video, path, fps=fps)
        logger.info(f"Video saved at {path}")

        save_copy_with_timestamp(path)
        return path

    def save_video_url(self, url):
        path = self.data.output_video_path

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
            save_copy_with_timestamp(path)
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

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

    def load_image(self, division=8, scale=1.0):
        if not os.path.exists(self.data.input_image_path):
            raise FileNotFoundError(self.data.input_image_path)

        image = load_image(self.data.input_image_path)

        logger.info(f"Image loaded from {self.data.input_image_path} size: {image.size}")
        self.orig_width, self.orig_height = image.size

        tmp = self.resize_image(image, division, scale)
        return tmp
