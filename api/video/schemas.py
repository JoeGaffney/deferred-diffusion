from pydantic import BaseModel


class VideoRequest(BaseModel):
    guidance_scale: float = 10.0
    input_image_path: str = "../tmp/input.png"
    max_height: int = 2048
    max_width: int = 2048
    model: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_frames: int = 48
    num_inference_steps: int = 25
    output_video_path: str = "../tmp/outputs/processed.mp4"
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5


class VideoResponse(BaseModel):
    data: str
