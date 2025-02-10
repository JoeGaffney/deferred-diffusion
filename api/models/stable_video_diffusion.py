import torch
from diffusers import StableVideoDiffusionPipeline
from api.common.context import Context
from api.utils import device_info

model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()


def main(context: Context):
    print("loading image")
    image = context.load_image()
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        image=image,
        num_inference_steps=context.num_inference_steps,
        num_frames=context.num_frames,
        decode_chunk_size=8,
        generator=generator,
    ).frames[0]

    processed_path = context.save_video(video)


if __name__ == "__main__":
    main(
        Context(
            image="space_v001.jpg",
            output_name="space",
            prompt="Detailed, 8k, add a spaceship, higher contrast, enchance keep original elements",
            num_frames=24,
            num_inference_steps=10,
        )
    )
    # main(
    #     Context(
    #         image="tornado_v001.jpg",
    #         strength=0.2,
    #         prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
    #         # num_inference_steps=50,
    #         size_multiplier=0.33,
    #     )
    # )
