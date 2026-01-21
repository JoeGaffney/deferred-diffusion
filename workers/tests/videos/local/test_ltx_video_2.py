import os

import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from transformers import (
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TorchAoConfig,
)


def get_pipeline(model_id) -> LTX2Pipeline:
    quant_config = TorchAoConfig("int8_weight_only")

    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    # Load pipeline with quantized components
    pipe = LTX2Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    # pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    return pipe


def test_text_to_video():
    width = 768
    height = 512
    frame_rate = 24.0
    random_seed = 42
    num_frames = 64
    generator = torch.Generator("cuda").manual_seed(random_seed)
    prompt = "a dog dancing to energetic electronic dance music"
    negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

    pipe = get_pipeline("Lightricks/LTX-2")

    video, audio = pipe.__call__(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        generator=generator,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=40,
        guidance_scale=4.0,
        output_type="np",
        return_dict=False,
    )

    video = (video * 255).round().astype("uint8")  # Type: ignore
    video = torch.from_numpy(video)

    encode_video(
        video[0],
        fps=int(frame_rate),
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path="/STORAGE/output/ltx2_distilled_sample_alt.mp4",
    )

    assert os.path.exists("/STORAGE/output/ltx2_distilled_sample_alt.mp4")
