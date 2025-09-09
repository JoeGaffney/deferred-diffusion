import os

import psutil
import pytest
import torch

from common.text_encoders import (
    get_pipeline_flux_text_encoder,
    get_pipeline_ltx_text_encoder,
    get_pipeline_qwen_text_encoder,
    get_pipeline_sd3_text_encoder,
    get_pipeline_wan_text_encoder,
)
from tests.constants import IMAGE_TO_IMAGE_PROMPT, TEXT_TO_IMAGE_PROMPT

torch_dtype = torch.float32

prompts = [TEXT_TO_IMAGE_PROMPT, IMAGE_TO_IMAGE_PROMPT]


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
    flux = get_pipeline_flux_text_encoder(torch_dtype=torch_dtype, device="cpu")
    report_cpu_memory("FLUX LOAD:")
    assert flux is not None
    flux_out = flux.encode(prompt)
    assert flux_out is not None
    report_cpu_memory("FLUX ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_sd3_text_encoder_cpu(prompt):
    sd3 = get_pipeline_sd3_text_encoder(torch_dtype=torch_dtype, device="cpu")
    report_cpu_memory("SD3 LOAD:")
    assert sd3 is not None
    sd3_out = sd3.encode(prompt)
    assert sd3_out is not None
    report_cpu_memory("SD3 ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_ltx_text_encoder_cpu(prompt):
    ltx = get_pipeline_ltx_text_encoder(torch_dtype=torch_dtype, device="cpu")
    report_cpu_memory("LTX LOAD:")
    assert ltx is not None
    ltx_out = ltx.encode(prompt)
    assert ltx_out is not None
    report_cpu_memory("LTX ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_qwen_text_encoder_cpu(prompt):
    qwen = get_pipeline_qwen_text_encoder(torch_dtype=torch_dtype, device="cpu")
    report_cpu_memory("QWEN LOAD:")
    assert qwen is not None
    qwen_out = qwen.encode(prompt)
    assert qwen_out is not None
    report_cpu_memory("QWEN ENCODE:")


@pytest.mark.parametrize("prompt", prompts)
def test_wan_text_encoder_cpu(prompt):
    wan = get_pipeline_wan_text_encoder(torch_dtype=torch_dtype, device="cpu")
    report_cpu_memory("WAN LOAD:")
    assert wan is not None
    wan_out = wan.encode(prompt)
    assert wan_out is not None
    report_cpu_memory("WAN ENCODE:")
