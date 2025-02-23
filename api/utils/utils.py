from datetime import datetime
import os
import shutil
from typing import Literal, Tuple
import numpy as np
import torch
from .logger import logger

Resolutions = Literal["1080p", "900p", "720p", "576p", "540p", "480p", "432p", "360p"]
resolutions_16_9 = {
    "1080p": (1920, 1080),  # by 8
    "900p": (1600, 900),
    "720p": (1280, 720),  # by 8
    "576p": (1024, 576),  # by 8 and 32
    "540p": (960, 540),
    "480p": (854, 480),
    "432p": (768, 432),  # by 8
    "360p": (640, 360),
}


def get_16_9_resolution(resolution: Resolutions) -> Tuple[int, int]:
    return resolutions_16_9.get(resolution, (960, 540))


def ensure_path_exists(path):
    my_dir = os.path.dirname(path)
    if not os.path.exists(my_dir):
        try:
            os.makedirs(my_dir)
        except Exception as e:
            print(e)
            pass


def save_copy_with_timestamp(path):
    if os.path.exists(path):
        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        # Create the timestamped path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Keep only 3 digits of milliseconds
        timestamp_path = os.path.join(directory, "tmp", f"{name}_{timestamp}{ext}")
        ensure_path_exists(timestamp_path)

        shutil.copy(path, timestamp_path)


# still some issues with this using as img2img
def encode_image_to_latents(image, vae):
    image = image.convert("RGB")  # Ensure no alpha channel
    image = np.array(image).astype(np.float16) / 255.0  # Normalize to [0,1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to("cuda")  # (H, W, C) → (1, C, H, W)
    image = (image - 0.5) * 2  # Normalize to [-1,1]

    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        latents = latent_dist.sample() * 0.18215

    return latents
