from transformers import T5EncoderModel, BitsAndBytesConfig


def optimize_pipeline(pipe, disable_safety_checker=True, enable_sequential_cpu_offload=False):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    # Enable CPU offload to save GPU memory
    pipe.enable_model_cpu_offload()

    # slower but allows larger SD3 models
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
    pipe.vae.enable_slicing()
    pipe.enable_attention_slicing("auto")  # Enable attention slicing for faster inference
    if disable_safety_checker:
        pipe.safety_checker = dummy_safety_checker

    return pipe


quantization_config = BitsAndBytesConfig(load_in_8bit=True)


def get_t5_quantized(model_id):

    return T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_3",
        quantization_config=quantization_config,
    )
