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
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        timestamp_path = os.path.join(directory, "tmp", f"{name}_{timestamp}{ext}")
        ensure_path_exists(timestamp_path)

        shutil.copy(path, timestamp_path)


def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
    return pixels


def vae_encode(image, vae):
    pixels = np.array(image)
    pixels = vae_encode_crop_pixels(pixels)
    print(f"Final shape after crop and batch dimension: {pixels.shape}")
    t = vae.encode(pixels[:, :, :, :3])
    return t
    # Convert the NumPy array directly to a PyTorch float16 tensor
    pixels_tensor = torch.from_numpy(pixels_result).half()  # Convert to float16 immediately

    # If the VAE model is on a GPU, move the tensor to the same device as the model
    # if next(vae.parameters()).is_cuda:
    #
    pixels_tensor = pixels_tensor.cuda()
    # Ensure that the input is in the correct shape (batch, channels, height, width)
    pixels_tensor = pixels_tensor.permute(
        0, 3, 1, 2
    )  # Reorder from (batch, height, width, channels) -> (batch, channels, height, width)

    # Use only the first 3 channels (RGB)
    latents = vae.encode(pixels_tensor[:, :3, :, :])  # Use only the first 3 channels (RGB)

    return latents
