import copy
import math
import tempfile
from typing import Literal

import requests
import torch
from diffusers.utils import export_to_video

from common.logger import logger
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    load_image_if_exists,
    load_video_frames_if_exists,
)
from videos.schemas import ModelName, VideoRequest


class VideoContext:
    def __init__(self, model: ModelName, data: VideoRequest):
        self.model = model
        self.data = data
        self.width = copy.copy(data.width)
        self.height = copy.copy(data.height)
        self.image = load_image_if_exists(data.image)
        if self.image:
            self.width, self.height = self.image.size

        self.video_frames = load_video_frames_if_exists(data.video)
        self.last_image = load_image_if_exists(data.last_image)

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def get_is_720p(self) -> bool:
        offset = 100
        # 720p is 1280x720 = 921,600 pixels total
        result = (self.width * self.height) >= (1280 * 720) - offset
        logger.info(f"Is 720p: {result} for dimensions {self.width}x{self.height}")
        return result

    def get_flow_shift(self) -> float:
        return 5.0 if self.get_is_720p() else 3.0

    def ensure_divisible(self, value: int):
        # Adjust width and height to be divisible by the specified value
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)
        if self.image:
            self.image = self.image.resize([self.width, self.height])
        if self.last_image:
            self.last_image = self.last_image.resize([self.width, self.height])

    def ensure_frames_divisible(self, current_frames, divisor: int = 4) -> int:
        return ((current_frames - 1) // divisor) * divisor + 1

    def get_divisible_num_frames(self, divisor: int = 4) -> int:
        current_frames = self.data.num_frames
        return self.ensure_frames_divisible(current_frames, divisor)

    def long_video(self) -> bool:
        return self.data.num_frames > 100

    def tmp_video_path(self):
        return tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".mp4").name

    def save_video(self, video, fps=24):
        path = self.tmp_video_path()
        path = export_to_video(video, output_video_path=path, fps=fps, quality=9)
        logger.info(f"Video saved at {path}")
        return path

    def save_video_url(self, url):
        path = self.tmp_video_path()

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path
