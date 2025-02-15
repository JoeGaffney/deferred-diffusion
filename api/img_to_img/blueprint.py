from flask import Blueprint, request, jsonify
from common.context import Context
from .models.stable_diffusion_xl_refine import main as stable_diffusion_xl_refine
from .models.stable_diffusion_xl_inpainting import main as stable_diffusion_xl_inpainting
from .models.stable_diffusion_3_5 import main as stable_diffusion_3_5
from .models.stable_diffusion_3_5_canny import main as stable_diffusion_3_5_canny

bp = Blueprint("img_to_img", __name__, url_prefix="/api")


@bp.route("img_to_img", methods=["POST"])
def img_to_img():
    data = request.json
    model = data.get("model")
    context = Context(
        input_image_path=data.get("input_image_path", "../tmp/input.png"),
        input_mask_path=data.get("input_mask_path", "../tmp/input_mask.png"),
        output_video_path=data.get("output_video_path", "../tmp/outputs/processed.mp4"),
        output_image_path=data.get("output_image_path", "../tmp/outputs/processed.png"),
        max_height=data.get("max_height", 2048),
        max_width=data.get("max_width", 2048),
        negative_prompt=data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
        num_frames=data.get("num_frames", 48),
        num_inference_steps=data.get("num_inference_steps", 25),
        prompt=data.get("prompt", "Detailed, 8k, photorealistic"),
        seed=data.get("seed", 42),
        strength=data.get("strength", 0.5),
        guidance_scale=data.get("guidance_scale", 10.0),
    )

    main = None
    if model == "stable_diffusion_xl_refine":
        main = stable_diffusion_xl_refine
    elif model == "stable_diffusion_xl_inpainting":
        main = stable_diffusion_xl_inpainting
    elif model == "stable_diffusion_3_5":
        main = stable_diffusion_3_5
    elif model == "stable_diffusion_3_5_canny":
        main = stable_diffusion_3_5_canny

    if not main:
        return jsonify({"error": "Invalid model"})

    result = main(context)

    return jsonify(result)
