from fastapi import APIRouter, HTTPException
from video.context import VideoContext
from video.models.cog_video_x import main as cog_video_x_main
from video.models.ltx_video import main as ltx_video_main
from video.models.runway_gen3 import main as runway_gen3_main
from video.models.stable_video_diffusion import main as stable_video_diffusion_main
from video.schemas import VideoRequest, VideoResponse

router = APIRouter(prefix="/video", tags=["Video"])


@router.post("/", response_model=VideoResponse)
def create(request: VideoRequest):
    context = VideoContext(request)

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
