import torch
from diffusers import LTXImageToVideoPipeline
from api.common.context import Context
from api.utils import device_info

model_id = "Lightricks/LTX-Video"
pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()


def main(context: Context):
    print("loading image")
    image = context.load_image()
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        num_frames=context.num_frames,
        generator=generator,
    ).frames[0]

    processed_path = context.save_video(video)


if __name__ == "__main__":
    main(
        Context(
            image="tornado_v001.jpg",
            strength=0.2,
            prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
            num_inference_steps=50,
            size_multiplier=0.3,
        )
    )
    # main(
    #     Context(
    #         image="earth_quake_v001.jpg",
    #         strength=0.2,
    #         prompt="Slow pan over a forest, trees falling, buildings shaking, Detailed, 8k, photorealistic, enchance keep original elements",
    #         num_inference_steps=50,
    #     )
    # )
