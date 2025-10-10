import os
import shutil

from tests.utils import image_to_base64, setup_output_file
from videos.context import VideoContext
from videos.schemas import ModelName, VideoRequest
from videos.tasks import model_router_main as main


def text_to_video(
    model: ModelName,
    prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    # prompt="An avalanche crashes down a mountain side. Thick torrential snow. Bright sunny day. Cinematic. Film quality.",
):
    output_name = setup_output_file(model, "text_to_video", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                prompt=prompt,
                num_frames=24,
            )
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
    # prompt="An avalanche crashes down a mountain side. Thick torrential snow. Bright sunny day. Cinematic. Film quality. People running away in panic.",
    prompt="A smoky, dimly lit dive bar at night. Charles Bukowski, rugged, unshaven, wearing a wrinkled shirt, sits hunched at the counter with a glass of whiskey in one hand and a cigarette smoldering in the other. The light flickers from a neon sign behind him. He slowly turns to the camera, raises his glass, and with a weary, sardonic smirk says: “Hello you original boozers. What matters most is how well you walk through the fire.” The atmosphere is gritty, raw, and cinematic, like a 1970s underground film.",
):
    output_name = setup_output_file(model, "text_to_video", extension="mp4")

    result = main(
        VideoContext(
            VideoRequest(
                model=model,
                prompt=prompt,
                num_frames=24,
                width=720,
                height=1280,
            )
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
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/color_v002.png"),
                prompt="A man with short gray hair plays a red electric guitar.",
                num_frames=24,
            )
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
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/wan_i2v_input.JPG"),
                prompt=prompt,
                num_frames=48,
            )
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
            VideoRequest(
                model=model,
                image=image_to_base64("../assets/act_char_v001.png"),
                video=image_to_base64("../assets/act_reference_v001.mp4"),
                prompt="A man in a tuxedo is waving at the camera.",
                num_inference_steps=6,
                num_frames=24,
            )
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
            VideoRequest(
                model=model,
                video=image_to_base64("../assets/act_reference_v001.mp4"),
            )
        )
    )

    if os.path.exists(result):
        shutil.copy(result, output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

    # Check if the output file is a valid video file
    assert os.path.getsize(output_name) > 100, f"Output file {output_name} is empty."
