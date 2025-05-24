import os

import torch
from transformers import BitsAndBytesConfig, TorchAoConfig

from common.logger import logger


def optimize_pipeline(pipe, disable_safety_checker=True, sequential_cpu_offload=False):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    # Enable CPU offload to save GPU memory
    if sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass  # VAE tiling is not available for all models

    # NOTE Breaks adapter workflows
    # pipe.enable_attention_slicing("auto")  # Enable attention slicing for faster inference
    if disable_safety_checker:
        pipe.safety_checker = dummy_safety_checker

    return pipe


def get_quant_dir(model_id: str, subfolder: str, load_in_4bit: bool) -> str:
    quant_bit = "4bit" if load_in_4bit else "8bit"
    subfolder_name = "default" if subfolder == "" else subfolder
    hf_home = os.getenv("HF_HOME", "")
    quant_dir = os.path.join(hf_home, "quantized", model_id, quant_bit, subfolder_name)
    return os.path.normpath(quant_dir)


def get_quantized_model(
    model_id,
    subfolder,
    model_class,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    disabled=False,
):
    """
    Load a quantized model component if available locally; otherwise, load original,
    quantize, save locally, and return.

    Args:
        model_id (str): Hugging Face repo/model ID.
        subfolder (str): Subfolder name for the model component (e.g., "transformer").
        model_class (class): The HF model class to load (e.g., WanTransformer3DModel).
        load_in_4bit (bool): Whether to load in 4-bit quantization. If False, loads in 8-bit.
        torch_dtype (torch.dtype): Dtype to use when loading.
        disabled (bool): If True, skip loading the quantized model.

    Returns:
        model instance
    """

    if disabled:
        logger.warning(f"Quantization disabled for {model_id} subfolder {subfolder}")
        return model_class.from_pretrained(model_id, subfolder=subfolder)

    quant_dir = get_quant_dir(model_id, subfolder, load_in_4bit)

    # NOTE does not support CPU model offload so using TorchAo for 8bit
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
    quant_config = TorchAoConfig("int8_weight_only")
    use_safetensors = False
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype
        )
        use_safetensors = True

    try:
        logger.info(f"Loading quantized model from {quant_dir}")
        model = model_class.from_pretrained(
            quant_dir, torch_dtype=torch_dtype, local_files_only=True, use_safetensors=use_safetensors
        )
    except Exception as e:
        logger.warning(f"Failed to load quantized model from {quant_dir}: {e}")
        logger.info(f"Loading and quantizing {model_id} subfolder {subfolder}")
        model = model_class.from_pretrained(
            model_id,
            subfolder=subfolder,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
        )
        os.makedirs(quant_dir, exist_ok=True)
        model.save_pretrained(quant_dir, safe_serialization=use_safetensors)
        logger.info(f"Saved quantized model to {quant_dir}")

    return model
