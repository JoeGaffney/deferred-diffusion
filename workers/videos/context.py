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
    def __init__(self, data: VideoRequest):
        self.model = data.model
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

    def get_resolution_type(self, offset: int = 100) -> Literal["480p", "720p", "1080p"]:
        pixels = self.width * self.height
        if pixels >= 1920 * 1080 - offset:
            return "1080p"  # 1080p is 1920x1080 = 2,073,600 pixels total
        elif pixels >= 1280 * 720 - offset:
            return "720p"  # 720p is 1280x720 = 921,600 pixels total
        return "480p"

    def get_flow_shift(self) -> float:
        return 5.0 if self.get_resolution_type() == "720p" else 3.0

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

    def duration_in_seconds(self, fps=24) -> int:
        return max(1, int(self.data.num_frames / fps))

    def tmp_video_path(self, model="") -> str:
        return tempfile.NamedTemporaryFile(dir=get_tmp_dir(model), suffix=".mp4").name

    def save_video(self, video, fps=24):
        path = self.tmp_video_path(self.model)
        path = export_to_video(video, output_video_path=path, fps=fps, quality=9)
        logger.info(f"Video saved at {path}")
        return path

    def save_video_url(self, url):
        path = self.tmp_video_path(model=self.model)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path
