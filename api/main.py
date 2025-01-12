from flask import Flask
from flask_socketio import SocketIO, emit
from io import BytesIO
import base64
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

@socketio.on('rendered_image')
def handle_rendered_image(data):
    # Decode the base64 image data
    image_data = data['data'].split(',')[1]
    decoded = base64.b64decode(image_data)
    input_image = Image.open(BytesIO(decoded)).convert("RGB")

    print(input_image)
    # Run Stable Diffusion (e.g., text-to-image or image-to-image)
    prompt = "A futuristic cityscape"
    processed_image = pipe(prompt, init_image=input_image, strength=0.5, guidance_scale=7.5).images[0]

    print(processed_image)
    # Convert to base64 to send back
    buffer = BytesIO()
    processed_image.save(buffer, format="PNG")
    processed_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    emit('processed_image', {'image': f'data:image/png;base64,{processed_image_base64}'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)