from datetime import datetime
import os
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from ..utils import device_info

model_id = "THUDM/CogVideoX-5b-I2V"
model_id = "THUDM/CogVideoX1.5-5b-I2V"
pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()


def save_with_timestamp(frames, prefix):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = "./tmp"
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_path = os.path.join(directory, f"{prefix}_image_{timestamp}.mp4")
    export_to_video(frames, image_path, fps=7)


def main(path="", strength=0.5, prompt="Detailed, 8k", size_multiplier=0.5):
    image = load_image(path)
    image = image.resize((512, 288))
    generator = torch.Generator(device="cuda").manual_seed(42)

    video = pipe.__call__(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=20,
        num_frames=10,
        guidance_scale=6,
        generator=generator,
    ).frames[0]

    save_with_timestamp(video, "processed")


if __name__ == "__main__":
    # main(
    #     path="../../tmp/space_v001.jpg",
    #     strength=0.4,
    #     prompt="Detailed",
    # )
    main(
        path="../../tmp/tornado_v001.jpg",
        strength=0.1,
        prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
        size_multiplier=1.0,
    )
