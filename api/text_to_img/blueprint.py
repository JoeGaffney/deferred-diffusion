from flask import Blueprint, request, jsonify
from common.context import Context
from models.auto_diffusion import main as auto_diffusion

bp = Blueprint("text_to_img", __name__, url_prefix="/api")


@bp.route("text_to_img", methods=["POST"])
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
    main = auto_diffusion

    if not main:
        return jsonify({"error": f"Invalid model {model}"})

    result = main(context, model_id=model, mode="text_to_image")

    return jsonify(result)
