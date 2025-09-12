from typing import List

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName, References
from images.tasks import model_router_main as main
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution


def text_to_image(model: ModelName, seed=42):
    output_name = setup_output_file(model, "text_to_image", suffix=str(seed))
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature.",
                strength=0.5,
                width=width,
                height=height,
                seed=seed,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def image_to_image(model: ModelName):
    output_name = setup_output_file(model, "image_to_image")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Change to night time and add rain and lighting",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def inpainting(model: ModelName):
    output_name = setup_output_file(model, "inpainting")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Photorealistic landscape of an elven castle, inspired by lord of the rings, highly detailed, 8k",
                strength=0.5,
                image=image_to_base64("../assets/inpaint.png"),
                mask=image_to_base64("../assets/inpaint_mask.png"),
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def inpainting_alt(model: ModelName):
    output_name = setup_output_file(model, "inpainting", suffix="_alt")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a tiger sitting on a park bench",
                strength=0.9,
                image=image_to_base64("../assets/inpaint_v002.png"),
                mask=image_to_base64("../assets/inpaint_mask_v002.png"),
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def references_canny(model: ModelName):
    output_name = setup_output_file(model, "references", "_canny")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Detailed, 8k, DSLR photo, photorealistic, eye",
                strength=0.5,
                width=width,
                height=height,
                references=[
                    References(
                        mode="canny",
                        image=image_to_base64("../assets/canny_v001.png"),
                        strength=0.5,
                    )
                ],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def references_style(model: ModelName):
    output_name = setup_output_file(model, "references", "_style")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a cat, masterpiece, best quality, high quality",
                strength=0.75,
                width=width,
                height=height,
                references=[
                    References(
                        mode="style",
                        image=image_to_base64("../assets/style_v001.jpeg"),
                        strength=0.5,
                    )
                ],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


def references_face(model: ModelName):
    """Test models with face adapter."""
    output_name = setup_output_file(model, "references", "_face")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a man walking, masterpiece, best quality, high quality",
                strength=0.75,
                width=width,
                height=height,
                references=[
                    References(
                        mode="style",
                        image=image_to_base64("../assets/style_v001.jpeg"),
                        strength=0.5,
                    ),
                    References(
                        mode="face",
                        image=image_to_base64("../assets/face_v001.jpeg"),
                        strength=0.5,
                    ),
                ],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)
