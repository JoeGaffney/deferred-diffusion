import importlib
import os
import shutil
from typing import Dict, Tuple

from tests.utils import image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest


def main(context: VideoContext):
    """Route to the specific model implementation by concrete model name.

    Lazy-imports the module/attribute that the corresponding celery task would call.
    """
    model = context.model

    MODEL_NAME_TO_CALLABLE: Dict[ModelName, Tuple[str, str]] = {
        "ltx-video": ("videos.models.ltx", "main"),
        "wan-2": ("videos.models.wan", "main"),
        # external implementations (match celery task targets)
        "runway-gen-4": ("videos.external_models.runway", "main"),
        "runway-act-two": ("videos.external_models.runway_act", "main"),
        "runway-upscale": ("videos.external_models.runway_upscale", "main"),
        "runway-gen-4-aleph": ("videos.external_models.runway_aleph", "main"),
        "bytedance-seedance-1": ("videos.external_models.bytedance_seedance", "main"),
        "kwaivgi-kling-2": ("videos.external_models.kling", "main"),
        "google-veo-3": ("videos.external_models.google_veo", "main"),
        "openai-sora-2": ("videos.external_models.openai", "main"),
    }

    if model not in MODEL_NAME_TO_CALLABLE:
        raise ValueError(f"No direct model implementation mapped for model '{model}'")

    module_path, attr = MODEL_NAME_TO_CALLABLE[model]
    mod = importlib.import_module(module_path)
    main_fn = getattr(mod, attr)
    return main_fn(context)


def text_to_video(
    model: ModelName,
    prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
):
    output_name = setup_output_file(model, "text_to_video", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                prompt=prompt,
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def text_to_video_portrait(
    model: ModelName,
    prompt="An avalanche crashes down a mountain side. Thick torrential snow. Bright sunny day. Cinematic. Film quality. People running away in panic.",
):
    output_name = setup_output_file(model, "text_to_video_portrait", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                prompt=prompt,
                num_frames=24,
                width=720,
                height=1280,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def image_to_video(model: ModelName):
    output_name = setup_output_file(model, "image_to_video", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                image=image_to_base64("../assets/color_v002.png"),
                prompt="A man with short gray hair plays a red electric guitar.",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def image_to_video_portrait(model):
    output_name = setup_output_file(model, "image_to_video", extension="mp4", suffix="portrait")
    prompt = "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat’s face, then cat resurfaces, still filming selfie, playful summer vacation mood."

    result = main(
        VideoContext(
            model,
            VideoRequest(
                image=image_to_base64("../assets/wan_i2v_input.JPG"),
                prompt=prompt,
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def video_to_video(model: ModelName):
    output_name = setup_output_file(model, "video_to_video", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                image=image_to_base64("../assets/act_char_v001.png"),
                video=image_to_base64("../assets/act_reference_v001.mp4"),
                prompt="A man in a tuxedo is waving at the camera.",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def video_upscale(model: ModelName):
    output_name = setup_output_file(model, "video_upscale", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                video=image_to_base64("../assets/act_reference_v001.mp4"),
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."


def first_frame_last_frame(model: ModelName):
    output_name = setup_output_file(model, "first_frame_last_frame", extension="mp4")

    result = main(
        VideoContext(
            model,
            VideoRequest(
                image=image_to_base64("../assets/first_frame_v001.png"),
                image_last_frame=image_to_base64("../assets/last_frame_v001.png"),
                prompt="The camera tracks into the man from behind the man is static",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
