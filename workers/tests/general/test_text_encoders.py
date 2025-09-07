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

torch_dtype = torch.bfloat16


def report_cpu_memory(prefix: str = ""):
    """Print current process CPU memory (RSS) in MB."""
    try:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        mb = rss / (1024 * 1024)
        print(f"{prefix} CPU memory RSS: {mb:.1f} MB")
    except Exception as e:
        print(f"Could not determine CPU memory: {e}")


def test_wan_text_encoder_cpu():
    wan = get_pipeline_wan_text_encoder(torch_dtype=torch_dtype, device="cpu")
    assert wan is not None
    wan_out = wan.encode("A short test prompt for WAN")
    assert wan_out is not None
    report_cpu_memory("WAN:")


def test_flux_text_encoder_cpu():
    flux = get_pipeline_flux_text_encoder(torch_dtype=torch_dtype, device="cpu")
    assert flux is not None
    flux_out = flux.encode("A short test prompt for FLUX")
    assert flux_out is not None
    report_cpu_memory("FLUX:")


def test_sd3_text_encoder_cpu():
    sd3 = get_pipeline_sd3_text_encoder(torch_dtype=torch_dtype, device="cpu")
    assert sd3 is not None
    sd3_out = sd3.encode("A short test prompt for SD3")
    assert sd3_out is not None
    report_cpu_memory("SD3:")


def test_ltx_text_encoder_cpu():
    ltx = get_pipeline_ltx_text_encoder(torch_dtype=torch_dtype, device="cpu")
    assert ltx is not None
    ltx_out = ltx.encode("A short test prompt for LTX")
    assert ltx_out is not None
    report_cpu_memory("LTX:")


def test_qwen_text_encoder_cpu():
    qwen = get_pipeline_qwen_text_encoder(torch_dtype=torch_dtype, device="cpu")
    assert qwen is not None
    qwen_out = qwen.encode("A short test prompt for QWEN")
    assert qwen_out is not None
    report_cpu_memory("QWEN:")
