from fastapi import APIRouter, HTTPException

from videos.context import VideoContext
from videos.models.cog_video_x import main as cog_video_x_main
from videos.models.ltx_video import main as ltx_video_main
from videos.models.runway_gen import main as runway_gen_main
from videos.models.stable_video_diffusion import main as stable_video_diffusion_main
from videos.schemas import VideoRequest, VideoResponse

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("", response_model=VideoResponse, operation_id="videos_create")
async def create(request: VideoRequest):
    context = VideoContext(request)

    main = None
    if request.model == "stable_video_diffusion":
        main = stable_video_diffusion_main
    elif request.model == "cog_video_x":
        main = cog_video_x_main
    elif request.model == "ltx_video":
        main = ltx_video_main
    elif request.model == "runway/gen3a_turbo" or request.model == "runway/gen4_turbo":
        main = runway_gen_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return VideoResponse(data=result)
