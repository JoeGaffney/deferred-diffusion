from fastapi import APIRouter, HTTPException

from videos.context import VideoContext
from videos.models.hunyuan_video import main as hunyuan_video_main
from videos.models.ltx_video import main as ltx_video_main
from videos.models.runway_gen import main as runway_gen_main
from videos.models.wan_2_1 import main as wan_2_1_main
from videos.schemas import VideoRequest, VideoResponse

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("", response_model=VideoResponse, operation_id="videos_create")
async def create(request: VideoRequest):
    context = VideoContext(request)

    main = None
    if request.model == "LTX-Video":
        main = ltx_video_main
    elif request.model == "HunyuanVideo":
        main = hunyuan_video_main
    elif request.model == "Wan2.1":
        main = wan_2_1_main
    elif request.model == "runway/gen3a_turbo" or request.model == "runway/gen4_turbo":
        main = runway_gen_main

    if not main:
        raise HTTPException(status_code=400, detail=f"Invalid model {request.model}")

    result = main(context)
    return VideoResponse(data=result)
