import gc

import pytest
import torch
from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, TorchAoConfig
from transformers import Qwen2_5_VLForConditionalGeneration


@pytest.fixture(
    params=[
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        TorchAoConfig("int8_weight_only"),
    ]
)
def quant_config(request):
    return request.param


def print_gpu_memory_usage(prefix=""):
    torch.cuda.reset_peak_memory_stats()
    mem_alloc = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    print(f"{prefix} Allocated: {mem_alloc / 1e6:.1f} MB, Reserved: {mem_reserved / 1e6:.1f} MB")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantized_model_warmup(quant_config):
    model_id = "black-forest-labs/FLUX.1-dev"
    torch_dtype = torch.bfloat16

    model = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", quantization_config=quant_config, torch_dtype=torch_dtype
    )
    print(str(quant_config))
    print_gpu_memory_usage("Before moving to GPU")

    # text_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", quantization_config=quant_config, torch_dtype=torch_dtype
    # )
    # print_gpu_memory_usage("After loading text model")

    model.to("cuda")
    print_gpu_memory_usage("After moving to GPU")

    mem_alloc = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    assert mem_alloc > 0
    assert mem_reserved > 0

    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print_gpu_memory_usage("After moving to CPU")
