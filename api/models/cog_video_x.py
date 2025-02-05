import torch
from diffusers import CogVideoXImageToVideoPipeline
from api.common.context import Context
from api.utils import device_info
from diffusers.utils import load_image

model_id = "THUDM/CogVideoX-5b-I2V"
model_id = "THUDM/CogVideoX1.5-5b-I2V"
pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()


def main(context: Context):
    print("loading image")
    image = load_image(context.get_input_image_path())
    image = image.resize((512, 288))

    print("processing generotor", image.size)
    generator = torch.Generator(device="cuda").manual_seed(42)
    print("processing image")

    video = pipe.__call__(
        width=1360,
        height=768,
        prompt=context.prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=20,
        num_frames=24,
        guidance_scale=6,
        generator=generator,
    ).frames[0]
    print("completed image")

    processed_image_path = context.save_video(video)


if __name__ == "__main__":
    main(
        Context(
            image="tornado_v001.jpg",
            strength=0.2,
            prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
            size_multiplier=1.0,
        )
    )
