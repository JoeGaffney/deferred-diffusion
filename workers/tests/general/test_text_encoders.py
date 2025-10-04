import os

import psutil
import pytest
import torch

from common.text_encoders import (
    flux_encode,
    ltx_encode,
    qwen_encode,
    sd3_encode,
    wan_encode,
)

TEXT_TO_IMAGE_PROMPT = "A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature."
IMAGE_TO_IMAGE_PROMPT = "Change to night time and add rain and lighting"

torch_dtype = torch.float32

prompts = [IMAGE_TO_IMAGE_PROMPT, IMAGE_TO_IMAGE_PROMPT]


def report_cpu_memory(prefix: str = ""):
    """Print current process CPU memory (RSS) in MB."""
    try:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        mb = rss / (1024 * 1024)
        print(f"{prefix} CPU memory RSS: {mb:.1f} MB")
    except Exception as e:
        print(f"Could not determine CPU memory: {e}")


@pytest.mark.parametrize("prompt", prompts)
def test_flux_text_encoder_cpu(prompt):
    flux_out = flux_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert flux_out is not None
    report_cpu_memory("FLUX ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_sd3_text_encoder_cpu(prompt):
    sd3_out = sd3_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert sd3_out is not None
    report_cpu_memory("SD3 ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_ltx_text_encoder_cpu(prompt):
    ltx_out = ltx_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert ltx_out is not None
    report_cpu_memory("LTX ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_qwen_text_encoder_cpu(prompt):
    qwen_out = qwen_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert qwen_out is not None
    report_cpu_memory("QWEN ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_wan_text_encoder_cpu(prompt):
    wan_out = wan_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert wan_out is not None
    report_cpu_memory("WAN ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_flux_is_still_in_cache(prompt):
    flux_out = flux_encode(prompt, torch_dtype=torch_dtype, device="cpu")
    assert flux_out is not None
    report_cpu_memory("FLUX CACHED ENCODE:")
