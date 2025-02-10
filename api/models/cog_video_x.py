import torch
from diffusers import CogVideoXImageToVideoPipeline
from common.context import Context

pipe = None


def get_pipeline():
    global pipe
    if pipe is None:
        model_id = "THUDM/CogVideoX1.5-5b-I2V"
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image()
    generator = torch.Generator(device="cuda").manual_seed(42)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        num_frames=context.num_frames,
        generator=generator,
        num_videos_per_prompt=1,
    ).frames[0]

    processed_image_path = context.save_video(video)


if __name__ == "__main__":
    main(
        Context(
            image="tornado_v001.jpg",
            strength=0.2,
            prompt="Detailed, 8k, photorealistic, wind, grass blowing in the wind, enchance keep original elements",
            # size_multiplier=0.33,
            # num_inference_steps=10,
        )
    )
