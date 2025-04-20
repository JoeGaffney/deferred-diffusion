import os

import pytest
import torch
from diffusers import AutoPipelineForText2Image

from common.pipeline_helpers import optimize_pipeline
from image.context import ImageContext
from image.models.auto_diffusion import main
from image.schemas import ImageRequest, IpAdapterModel
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["text_to_image", "img_to_img"]

MODELS = [
    "sd1.5",
    "sdxl",
]

MODEL_STYLE_ADAPTERS_MAPPING = {
    "sd1.5": [
        IpAdapterModel(
            image_path="../test_data/style_v001.jpeg",
            model="h94/IP-Adapter",
            scale=0.5,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        ),
    ],
    "sdxl": [
        IpAdapterModel(
            image_path="../test_data/style_v001.jpeg",
            model="h94/IP-Adapter",
            scale=0.5,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        ),
    ],
}

MODEL_FACE_ADAPTERS_MAPPING = {
    "sdxl": [
        IpAdapterModel(
            image_path="../test_data/style_v001.jpeg",
            model="h94/IP-Adapter",
            scale=0.5,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.bin",
            image_encoder=True,
        ),
        IpAdapterModel(
            image_path="../test_data/face_v001.jpeg",
            model="h94/IP-Adapter",
            scale=0.5,
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
            image_encoder=True,
        ),
    ],
}


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
    ip_adapters = MODEL_STYLE_ADAPTERS_MAPPING[model_id]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="" if mode == "text_to_image" else "../test_data/color_v001.jpeg",
                input_mask_path="../test_data/mask_v001.png",
                output_image_path=output_name,
                prompt="a cat, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=ip_adapters,
            )
        ),
        mode=mode,
    )

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", ["sdxl"])
def test_face(model_id, mode):
    """Test models with face adapter."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_face_adapter.png"
    width, height = get_16_9_resolution("540p")
    ip_adapters = MODEL_FACE_ADAPTERS_MAPPING[model_id]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="" if mode == "text_to_image" else "../test_data/color_v001.jpeg",
                input_mask_path="../test_data/mask_v001.png",
                output_image_path=output_name,
                prompt="a man walking, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=ip_adapters,
            )
        ),
        mode=mode,
    )

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."
