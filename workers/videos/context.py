import copy
import tempfile
from pathlib import Path
from typing import Literal

import requests
import torch
from diffusers.utils import export_to_video

from common.config import settings
from common.logger import get_task_id, logger, task_log
from utils.utils import (
    ensure_divisible,
    image_crop,
    image_resize,
    load_image_if_exists,
    load_video_frames_if_exists,
)
from videos.schemas import VideoRequest


class VideoContext:
    def __init__(self, data: VideoRequest, task_id: str | None = None):
        self.model = data.model
        self.task_id = task_id or get_task_id()
        self.data = data
        self.width = copy.copy(data.width)
        self.height = copy.copy(data.height)
        self.image = load_image_if_exists(data.image)
        if self.image:
            self.width, self.height = self.image.size

        self.video_frames = load_video_frames_if_exists(data.video, model=self.model)
        self.last_image = load_image_if_exists(data.last_image)

        task_log(
            f"Context created {self.model}, {self.width}x{self.height}",
        )

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_mega_pixels(self) -> float:
        """Calculate the megapixels based on width and height.
        480p (854x480): ~0.41 MP
        580p (1024x576): ~0.59 MP
        720p (1280x720): ~0.92 MP
        1080p (1920x1080): ~2.07 MP
        1440p (2560x1440): ~3.69 MP
        4K (3840x2160): ~8.29 MP
        """
        return (self.width * self.height) / 1_000_000

    def rescale_to_max_megapixels(self, max_mp: float):
        """Rescale width and height to ensure the total megapixels do not exceed max_mp."""
        current_mp = self.get_mega_pixels()
        if current_mp <= max_mp:
            return  # Already under the limit

        scale_factor = (max_mp / current_mp) ** 0.5  # sqrt because area scales with both width and height
        self.width = int(self.width * scale_factor)
        self.height = int(self.height * scale_factor)

        if self.image:
            self.image = image_resize(self.image, (self.width, self.height))
        if self.last_image:
            self.last_image = image_resize(self.last_image, (self.width, self.height))
        if self.video_frames:
            for frame_idx in range(len(self.video_frames)):
                self.video_frames[frame_idx] = image_resize(self.video_frames[frame_idx], (self.width, self.height))

    def ensure_divisible(self, value: int):
        """Adjust width and height to be divisible by the specified value."""
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)

        if self.image:
            self.image = image_crop(self.image, (self.width, self.height))
        if self.last_image:
            self.last_image = image_crop(self.last_image, (self.width, self.height))
        if self.video_frames:
            for frame_idx in range(len(self.video_frames)):
                self.video_frames[frame_idx] = image_crop(self.video_frames[frame_idx], (self.width, self.height))

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def ensure_frames_divisible(self, current_frames, divisor: int = 4) -> int:
        return ((current_frames - 1) // divisor) * divisor + 1

    def duration_in_seconds(self, fps=24) -> int:
        return max(1, int(self.data.num_frames / fps))

    def get_output_path(self, index: int = 0) -> Path:
        # deterministic relative path
        task_dir = self.data.task_name.replace(".", "/")
        rel_path = Path(task_dir) / f"{self.task_id}-{index}.mp4"
        abs_path = Path(settings.storage_dir / rel_path).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path

    def save_output(self, video, index: int = 0, fps=24) -> Path:
        abs_path = self.get_output_path(index)
        try:
            export_to_video(video, output_video_path=str(abs_path), fps=fps, quality=9)
            logger.info(f"Video saved at {abs_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save video at {abs_path}: {e}")

        return abs_path

    def save_output_url(self, url, index: int = 0) -> Path:
        abs_path = self.get_output_path(index)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        try:
            with open(abs_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            logger.info(f"Video saved at {abs_path}")
        except Exception as e:
            raise Exception(f"Failed to download or write file") from e

        return abs_path
