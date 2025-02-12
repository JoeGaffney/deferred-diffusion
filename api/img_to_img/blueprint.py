from flask import Blueprint, request, jsonify
from common.context import Context
from .models.stable_diffusion_xl_refine import main as stable_diffusion_xl_refine_main

bp = Blueprint("img_to_img", __name__, url_prefix="/api")


@bp.route("img_to_img", methods=["POST"])
def img_to_img():
    data = request.json
    model = data.get("model")
    context = Context(
        image=data.get("image", ""),
        input_dir=data.get("input_dir", "../tmp"),
        max_height=data.get("max_height", 2048),
        max_width=data.get("max_width", 2048),
        negative_prompt=data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
        num_frames=data.get("num_frames", 48),
        num_inference_steps=data.get("num_inference_steps", 25),
        output_dir=data.get("output_dir", "../tmp/outputs"),
        output_name=data.get("output_name", "processed"),
        prompt=data.get("prompt", "Detailed, 8k, photorealistic"),
        seed=data.get("seed", 42),
        strength=data.get("strength", 0.5),
    )

    main = None
    if model == "stable_diffusion_xl_refine":
        main = stable_diffusion_xl_refine_main

    if not main:
        return jsonify({"error": "Invalid model"})

    result = main(context)

    return jsonify(result)
