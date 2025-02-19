import torch
from diffusers import CogVideoXImageToVideoPipeline
from common.context import Context
from utils.utils import get_16_9_resolution

pipe = None

model_id = "THUDM/CogVideoX1.5-5b-I2V"
# model_id = "NimVideo/cogvideox-2b-img2vid"


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image(division=16)
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

    processed_image_path = context.save_video(video, fps=16)
    return processed_image_path


if __name__ == "__main__":
    # width, height = get_16_9_resolution("432p")
    width, height = get_16_9_resolution("540p")
    main(
        Context(
            input_image_path="../tmp/tornado_v001.jpg",
            output_video_path="../tmp/output/tornado_v001_cog_video_x.mp4",
            strength=0.2,
            prompt="Tornado spinning in a farm land",
            negative_prompt="blurry, distorted",
            num_inference_steps=40,
            max_width=width,
            max_height=height,
            num_frames=81,
        )
    )
