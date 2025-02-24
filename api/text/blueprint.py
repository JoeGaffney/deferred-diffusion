from flask import Blueprint, request, jsonify
from common.context import Context
from text.models.qwen_2_5_vl_instruct import main as qwen_2_5_vl_instruct_main

bp = Blueprint("text", __name__, url_prefix="/api")


@bp.route("text", methods=["POST"])
def diffusion():
    data = request.json
    model = data.get("model", "qwen_2_5_vl_instruct")
    context = Context(
        negative_prompt=data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
        num_frames=data.get("num_frames", 48),
        prompt=data.get("prompt", "Detailed, 8k, photorealistic"),
        seed=data.get("seed", 42),
        model=model,
        messages=data.get("messages", []),
    )
    main = None
    if model == "qwen_2_5_vl_instruct":
        main = qwen_2_5_vl_instruct_main

    if not main:
        return jsonify({"error": f"Invalid model {str(model)}"})

    result = main(context)

    return jsonify(result)
