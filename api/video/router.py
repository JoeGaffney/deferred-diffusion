from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from common.context import Context
from video.models.stable_video_diffusion import main as stable_video_diffusion_main
from video.models.cog_video_x import main as cog_video_x_main
from video.models.ltx_video import main as ltx_video_main
from video.models.runway_gen3 import main as runway_gen3_main

router = APIRouter(prefix="/video", tags=["Video"])


class VideoRequest(BaseModel):
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


@router.post("/", response_model=VideoResponse)
def create(request: VideoRequest):
    context = Context(
        input_image_path=request.input_image_path,
        output_video_path=request.output_video_path,
        max_height=request.max_height,
        max_width=request.max_width,
        negative_prompt=request.negative_prompt,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        prompt=request.prompt,
        seed=request.seed,
        strength=request.strength,
    )

    main = None
    if request.model == "stable_video_diffusion":
        main = stable_video_diffusion_main
    elif request.model == "cog_video_x":
        main = cog_video_x_main
    elif request.model == "ltx_video":
        main = ltx_video_main
    elif request.model == "runway/gen3a_turbo":
        main = runway_gen3_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return VideoResponse(data=result)
