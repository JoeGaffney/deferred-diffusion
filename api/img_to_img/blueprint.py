from flask import Blueprint, request, jsonify
from common.context import Context
from .models.stable_diffusion_upscaler import main as stable_diffusion_upscaler
from .models.auto_diffusion import main as auto_diffusion

bp = Blueprint("img_to_img", __name__, url_prefix="/api")


@bp.route("img_to_img", methods=["POST"])
def img_to_img():
    data = request.json
    model = data.get("model")
    context = Context(
        input_image_path=data.get("input_image_path", "../tmp/input.png"),
        input_mask_path=data.get("input_mask_path", ""),
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
        inpainting_full_image=data.get("inpainting_full_image", True),
        disable_text_encoder_3=data.get("disable_text_encoder_3", True),
        controlnets=data.get("controlnets", []),
    )

    main = None
    if model == "stable_diffusion_upscaler":
        main = stable_diffusion_upscaler
        result = main(context, model_id=model, mode="upscaler")
        return jsonify(result)

    # vary the mode based on the inputs
    mode = "img_to_img"
    if context.input_mask_path != "":
        mode = "img_to_img_inpainting"

    # does not support inpainting
    if model == "stabilityai/stable-diffusion-xl-refiner-1.0":
        mode = "img_to_img"

    if context.input_image_path == "":
        mode = "text_to_image"

    main = auto_diffusion

    if not main:
        return jsonify({"error": "Invalid model"})

    result = main(context, model_id=model, mode=mode)
    return jsonify(result)
