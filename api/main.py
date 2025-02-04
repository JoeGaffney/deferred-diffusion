import math
from flask import Flask
from flask_socketio import SocketIO, emit
from io import BytesIO
import base64
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image, StableDiffusionDepth2ImgPipeline
from diffusers.utils import make_image_grid, load_image
import os
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

# pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-depth",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
# ).to("cuda")

pipe.enable_model_cpu_offload()

# Override the safety checker
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

def save_image_with_timestamp(image, prefix):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = "./tmp"
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_path = os.path.join(directory, f"{prefix}_image_{timestamp}.png")
    image.save(image_path)
    return image_path


@socketio.on('rendered_image')
def handle_rendered_image(data):
    # Decode the base64 image data
    image_data = data['data'].split(',')[1]
    decoded = base64.b64decode(image_data)
    input_image = Image.open(BytesIO(decoded)).convert("RGB")

    # Resize the image
    size_multiplier = 0.5
    width = input_image.size[0] * size_multiplier
    height = input_image.size[1] * size_multiplier
    width = math.ceil(width / 8) * 8
    height = math.ceil(height / 8) * 8
    input_image = input_image.resize((width, height))

    # Flip the image horizontally
    input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the input image with a timestamp
    input_image_path = save_image_with_timestamp(input_image, "input")
    print(input_image, input_image_path)

    # Run Stable Diffusion (e.g., text-to-image or image-to-image)
    prompt = "A futuristic cityscape , detailed, 8k"
    # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    # prompt = "Update the image to be a photorealistic landscape, sunset color palette, contracting colors, detailed, 8k"

    # Calculate height and width based on the input image size and size multiplier
    # Ensure are divisible by 8
    # size_multiplier = 0.5
    # width = input_image.size[0] * size_multiplier
    # height = input_image.size[1] * size_multiplier
    # width = math.ceil(width / 8) * 8
    # height = math.ceil(height / 8) * 8

    # processed_image = pipe(prompt, seed=1000, height=height, width=width, image=input_image, strength=0.5, guidance_scale=7.5, num_inference_steps=50).images[0]
    
    # prepare image
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
    # init_image = load_image(url)
    processed_image = pipe(prompt, image=input_image, strength=0.5).images[0]


    # Save the processed image with a timestamp
    processed_image_path = save_image_with_timestamp(processed_image, "processed")
    print(processed_image, processed_image_path)
    # Convert to base64 to send back
    buffer = BytesIO()
    processed_image.save(buffer, format="PNG")
    processed_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    emit('processed_image', {'image': f'data:image/png;base64,{processed_image_base64}'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)