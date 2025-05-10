import os

import pytest
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from common.pipeline_helpers import optimize_pipeline
from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest, IpAdapterModel
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import ensure_path_exists, get_16_9_resolution

# Define constants
MODES = ["text_to_image", "img_to_img"]

MODELS = [
    "sd1.5",
    "sdxl",
]


@pytest.mark.skip(reason="debug only")
def test_load_adapter():
    """Test loading of style adapter."""
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline = optimize_pipeline(pipeline)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_style(model_id, mode):
    """Test models with style adapter."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_style_adapter.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../test_data/color_v001.jpeg"),
                prompt="a cat, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=[
                    IpAdapterModel(image=image_to_base64("../test_data/style_v001.jpeg"), model="style", scale=0.5)
                ],
            )
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", MODELS)
def test_face(model_id, mode):
    """Test models with face adapter."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_face_adapter.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                prompt="a man walking, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=[
                    IpAdapterModel(image=image_to_base64("../test_data/style_v001.jpeg"), model="style", scale=0.5),
                    IpAdapterModel(image=image_to_base64("../test_data/face_v001.jpeg"), model="face", scale=0.5),
                ],
            )
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."
