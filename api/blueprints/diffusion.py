from flask import Blueprint, request, jsonify
from api.common.context import Context
from models.stable_video_diffusion import main as stable_video_diffusion_main
from models.cog_video_x import main as cog_video_x_main
from models.ltx_video import main as ltx_video_main
from models.stable_diffusion_xl_refine import main as stable_diffusion_xl_refine_main

bp = Blueprint("diffusion", __name__, url_prefix="/api")


@bp.route("diffusion", methods=["POST"])
def diffusion():
    data = request.json
    model = data.get("model")
    context = Context(
        image=data.get("image", ""),
        input_dir=data.get("input_dir", "./tmp"),
        max_height=data.get("max_height", round(1080 * 0.5333)),
        max_width=data.get("max_width", round(1920 * 0.5333)),
        negative_prompt=data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
        num_frames=data.get("num_frames", 48),
        num_inference_steps=data.get("num_inference_steps", 25),
        output_dir=data.get("output_dir", "./tmp/outputs"),
        output_name=data.get("output_name", "processed"),
        prompt=data.get("prompt", "Detailed, 8k, photorealistic"),
        seed=data.get("seed", 42),
        size_multiplier=data.get("size_multiplier", 1.0),
        strength=data.get("strength", 0.5),
    )

    main = None
    if model == "stable_video_diffusion":
        main = stable_video_diffusion_main
    elif model == "cog_video_x":
        main = cog_video_x_main
    elif model == "ltx_video":
        main = ltx_video_main
    elif model == "stable_diffusion_xl_refine":
        main = stable_diffusion_xl_refine_main

    if not main:
        return jsonify({"error": "Invalid model"})

    result = main(context)

    return jsonify(result)
