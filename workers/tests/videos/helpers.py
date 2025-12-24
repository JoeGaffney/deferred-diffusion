import importlib
import os
import shutil
from typing import Dict, Tuple

from tests.utils import assert_file_exists, image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest


def main(context: VideoContext):
    """Route to the specific model implementation by concrete model name.

    Lazy-imports the module/attribute that the corresponding celery task would call.
    """
    model = context.model

    MODEL_NAME_TO_CALLABLE: Dict[ModelName, Tuple[str, str]] = {
        "ltx-video": ("videos.local.ltx_video", "main"),
        "wan-2": ("videos.local.wan_2", "main"),
        "hunyuan-video-1": ("videos.local.hunyuan_video_1", "main"),
        "sam-3": ("videos.local.sam_3", "main"),
        "runway-gen-4": ("videos.external.runway_gen_4", "main"),
        "runway-upscale": ("videos.external.runway_upscale", "main"),
        "seedance-1": ("videos.external.seedance_1", "main"),
        "kling-2": ("videos.external.kling_2", "main"),
        "veo-3": ("videos.external.veo_3", "main"),
        "sora-2": ("videos.external.sora_2", "main"),
        "hailuo-2": ("videos.external.hailuo_2", "main"),
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
    num_frames=24,
):
    output_name = setup_output_file(model, "text_to_video", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                prompt=prompt,
                num_frames=num_frames,
                width=int(1280 / 1.5),
                height=int(720 / 1.5),
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
            VideoRequest(
                model=model,
                prompt=prompt,
                num_frames=24,
                width=int(720 / 1.5),
                height=int(1280 / 1.5),
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def image_to_video(model: ModelName):
    output_name = setup_output_file(model, "image_to_video", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/color_v002.png"),
                prompt="A man with short gray hair plays a red electric guitar.",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def image_to_video_portrait(model):
    output_name = setup_output_file(model, "image_to_video", extension="mp4", suffix="portrait")
    prompt = "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat’s face, then cat resurfaces, still filming selfie, playful summer vacation mood."

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/wan_i2v_input.JPG"),
                prompt=prompt,
                num_frames=48,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def video_to_video(model: ModelName):
    output_name = setup_output_file(model, "video_to_video", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/act_char_v001.png"),
                video=image_to_base64("../assets/act_reference_v001.mp4"),
                prompt="A man in a tuxedo is waving at the camera.",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def video_upscale(model: ModelName):
    output_name = setup_output_file(model, "video_upscale", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                video=image_to_base64("../assets/act_reference_v001.mp4"),
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def first_frame_last_frame(model: ModelName):
    output_name = setup_output_file(model, "first_frame_last_frame", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/first_frame_v001.png"),
                last_image=image_to_base64("../assets/last_frame_v001.png"),
                prompt="a dramatic dolly zoom",
                num_frames=24,
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def video_segmentation(model: ModelName):
    output_name = setup_output_file(model, "video_segmentation", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(model=model, video=image_to_base64("../assets/act_reference_v001.mp4"), prompt="man"),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)


def video_segmentation_alt(model: ModelName):
    output_name = setup_output_file(model, "video_segmentation_alt", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model, video=image_to_base64("../assets/act_reference_v001.mp4"), prompt="man, hands, face"
            ),
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    assert_file_exists(output_name)
