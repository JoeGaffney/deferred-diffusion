import importlib
from pathlib import Path
from typing import Dict, List, Tuple

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName, References
from tests.utils import asset_outputs_exists, image_to_base64
from utils.utils import get_16_9_resolution


def main(context: ImageContext) -> List[Path]:
    """Route to the specific model implementation by concrete model name.

    Lazy-imports the module/attribute that the corresponding celery task would call.
    """
    model = context.model

    MODEL_NAME_TO_CALLABLE: Dict[ModelName, Tuple[str, str]] = {
        "sd-xl": ("images.local.sd_xl", "main"),
        "flux-1": ("images.local.flux_1", "main"),
        "flux-2": ("images.local.flux_2", "main"),
        "qwen-image": ("images.local.qwen_image", "main"),
        "z-image": ("images.local.z_image", "main"),
        "depth-anything-2": ("images.local.depth_anything_2", "main"),
        "sam-2": ("images.local.sam_2", "main"),
        "sam-3": ("images.local.sam_3", "main"),
        "real-esrgan-x4": ("images.local.real_esrgan_x4", "main"),
        "gpt-image-1": ("images.external.gpt_image_1", "main"),
        "runway-gen-4": ("images.external.runway_gen_4", "main"),
        "flux-1-pro": ("images.external.flux_1_pro", "main"),
        "flux-2-pro": ("images.external.flux_2_pro", "main"),
        "topazlabs-upscale": ("images.external.topazlabs_upscale", "main"),
        "gemini-2": ("images.external.gemini_2", "main"),
        "gemini-3": ("images.external.gemini_3", "main"),
        "seedream-4": ("images.external.seedream_4", "main"),
    }

    if model not in MODEL_NAME_TO_CALLABLE:
        raise ValueError(f"No direct model implementation mapped for model '{model}'")

    module_path, attr = MODEL_NAME_TO_CALLABLE[model]
    mod = importlib.import_module(module_path)
    main_fn = getattr(mod, attr)
    return main_fn(context)


def text_to_image(model: ModelName, seed=42):
    output_name = f"text_to_image_{str(seed)}"
    width, height = get_16_9_resolution("720p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature.",
                strength=0.5,
                width=width,
                height=height,
                seed=seed,
            ),
            task_id=output_name,
        )
    )
    asset_outputs_exists(result)


def text_to_image_alt(model: ModelName, seed=42):
    output_name = f"text_to_image_alt_{str(seed)}"
    width, height = get_16_9_resolution("720p")
    prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt=prompt,
                strength=0.5,
                width=width,
                height=height,
                seed=seed,
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def image_to_image(model: ModelName):
    output_name = "image_to_image"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Change to night time and add rain and Lightning",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def image_to_image_alt(model: ModelName):
    output_name = "image_to_image_alt"
    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Change the car color to red, turn the headlights on",
                strength=0.5,
                image=image_to_base64("../assets/color_v003.png"),
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def inpainting(model: ModelName):
    output_name = "inpainting"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Photorealistic landscape of an elven castle, inspired by lord of the rings, highly detailed, 8k",
                strength=0.5,
                image=image_to_base64("../assets/inpaint.png"),
                mask=image_to_base64("../assets/inpaint_mask.png"),
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def inpainting_alt(model: ModelName):
    output_name = "inpainting_alt"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a tiger sitting on a park bench",
                strength=0.9,
                image=image_to_base64("../assets/inpaint_v003.png"),
                mask=image_to_base64("../assets/inpaint_mask_v003.png"),
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def references_canny(model: ModelName):
    output_name = "references_canny"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="A close up of an eye, Detailed, 8k, DSLR photo, photorealistic",
                strength=0.5,
                width=1152,
                height=768,
                references=[
                    References(
                        image=image_to_base64("../assets/canny_v001.png"),
                    )
                ],
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def references_depth(model: ModelName):
    output_name = "references_depth"
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Two people hiking in a mountainous landscape, photorealistic high detail, 8k resolution, use the depth image for placement.",
                strength=0.5,
                width=width,
                height=height,
                references=[
                    References(
                        image=image_to_base64("../assets/depth_v001.png"),
                    )
                ],
            ),
        )
    )

    asset_outputs_exists(result)


def references_style(model: ModelName):
    output_name = "references_style"
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a cat walking, photorealistic, best quality, high quality",
                strength=0.75,
                width=width,
                height=height,
                references=[
                    References(
                        image=image_to_base64("../assets/style_v001.jpeg"),
                    )
                ],
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


def references_face(model: ModelName):
    output_name = "references_face"
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
                        image=image_to_base64("../assets/style_v001.jpeg"),
                    ),
                    References(
                        image=image_to_base64("../assets/face_v001.jpeg"),
                    ),
                ],
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)
