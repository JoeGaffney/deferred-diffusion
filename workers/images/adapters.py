from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image

from common.exceptions import IPAdapterConfigError
from common.logger import logger
from images.schemas import IpAdapterModel, IpAdapterModelConfig, ModelConfig
from utils.utils import load_image_if_exists

processor = IPAdapterMaskProcessor()

generic_ip_adapter_model = IpAdapterModelConfig(
    model="default",
    subfolder="default",
    weight_name="default",
    image_encoder=False,
    image_encoder_subfolder="default",
)

IP_ADAPTER_MODEL_CONFIG = {
    "sd1.5": {
        "style": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
        "style-plus": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
        "face": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
    },
    "sdxl": {
        "style": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
        "style-plus": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
        "face": IpAdapterModelConfig(
            model="h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
            image_encoder=True,
            image_encoder_subfolder="models/image_encoder",
        ),
    },
    "flux": {
        "style": IpAdapterModelConfig(
            model="XLabs-AI/flux-ip-adapter-v2",
            subfolder="default",
            weight_name="ip_adapter.safetensors",
            image_encoder=True,
            image_encoder_subfolder="openai/clip-vit-large-patch14",
        ),
        "style-plus": IpAdapterModelConfig(
            model="XLabs-AI/flux-ip-adapter-v2",
            subfolder="default",
            weight_name="ip_adapter.safetensors",
            image_encoder=True,
            image_encoder_subfolder="openai/clip-vit-large-patch14",
        ),
        "face": IpAdapterModelConfig(
            model="XLabs-AI/flux-ip-adapter-v2",
            subfolder="default",
            weight_name="ip_adapter.safetensors",
            image_encoder=True,
            image_encoder_subfolder="openai/clip-vit-large-patch14",
        ),
    },
    # openai and runway is a special case, it uses just the images and not the model - but we still use the same ipdapter flow for parity
    "openai": {
        "style": generic_ip_adapter_model,
        "style-plus": generic_ip_adapter_model,
        "face": generic_ip_adapter_model,
    },
    "runway": {
        "style": generic_ip_adapter_model,
        "style-plus": generic_ip_adapter_model,
        "face": generic_ip_adapter_model,
    },
}


def get_ip_adapter_config(model_family: str, adapter_type: str) -> IpAdapterModelConfig:
    model_config = IP_ADAPTER_MODEL_CONFIG.get(model_family)
    if not model_config:
        raise IPAdapterConfigError(f"IP-Adapter model config for {model_family} not found")

    ip_adapter_config = model_config.get(adapter_type)
    if not ip_adapter_config:
        raise IPAdapterConfigError(f"IP-Adapter model path for {adapter_type} not found in {model_family}")

    return ip_adapter_config


class IpAdapter:
    def __init__(self, data: IpAdapterModel, model_config: ModelConfig, width, height):
        self.config = get_ip_adapter_config(model_config.model_family, data.model)
        self.scale = data.scale
        self.scale_layers = data.scale_layers
        self.model = self.config.model
        if model_config.model_family == "flux":
            self.scale_layers = "default"

        self.image = load_image_if_exists(data.image)

        if not self.image:
            raise IPAdapterConfigError(f"Could not load IP-Adapter image from {data.image}")

        if self.scale < 0.01:
            raise IPAdapterConfigError("IP-Adapter scale must be >= 0.01")

        # we allways need a full mask if some use a mask
        self.mask_image = Image.new("L", (width, height), 255)  # Create 512x512 white image in L mode
        tmp_mask_image = load_image_if_exists(data.mask)
        if tmp_mask_image:
            self.mask_image = tmp_mask_image.resize([width, height])
            self.mask_image = self.mask_image.convert("L")

    def get_scale_layers(self):

        # NOTE see https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter#style--layout-control
        if self.scale_layers == "style":
            return {
                "up": {"block_0": [0.0, self.scale, 0.0]},
            }
        elif self.scale_layers == "style_and_layout":
            return {
                "down": {"block_2": [0.0, self.scale]},
                "up": {"block_0": [0.0, self.scale, 0.0]},
            }
        elif self.scale_layers == "layout":
            return {
                "down": {"block_2": [0.0, self.scale]},
            }

        return self.scale

    def get_mask(self):
        return processor.preprocess(self.mask_image)


class Adapters:
    def __init__(self, adapters: list[IpAdapterModel], model_config: ModelConfig, width, height):
        self.adapters: list[IpAdapter] = []

        # Handle initialization errors and create valid adapters
        for data in adapters:
            try:
                adapter = IpAdapter(data, model_config, width, height)
                self.adapters.append(adapter)
            except IPAdapterConfigError as e:
                logger.error(f"Failed to initialize IP-Adapter: {e}")

    def is_enabled(self) -> bool:
        """Check if there are any valid adapters."""
        return len(self.adapters) > 0

    def get_scales_and_layers(self):
        return [adapter.get_scale_layers() for adapter in self.adapters]

    def get_masks(self):
        return [adapter.get_mask() for adapter in self.adapters]

    def get_images(self):
        return [adapter.image for adapter in self.adapters]

    def get_pipeline_config(self):
        """Get the configuration needed for the pipeline."""
        if not self.is_enabled():
            return {
                "models": (),
                "subfolders": (),
                "weights": (),
                "image_encoder_model": "",
                "image_encoder_subfolder": "",
            }

        models = []
        subfolders = []
        weights = []
        image_encoder_model = ""
        image_encoder_subfolder = ""

        for adapter in self.adapters:
            models.append(adapter.config.model)
            subfolders.append(adapter.config.subfolder)
            weights.append(adapter.config.weight_name)
            if adapter.config.image_encoder:
                image_encoder_model = adapter.config.model
                image_encoder_subfolder = adapter.config.image_encoder_subfolder

        return {
            "models": tuple(models),
            "subfolders": tuple(subfolders),
            "weights": tuple(weights),
            "image_encoder_model": image_encoder_model,
            "image_encoder_subfolder": image_encoder_subfolder,
        }

    def set_scale(self, pipe):
        """Set the IP adapter scale on the pipeline."""
        if not self.is_enabled():
            return pipe

        scales = self.get_scales_and_layers()
        if len(scales) == 1:
            pipe.set_ip_adapter_scale(scales[0])
        else:
            pipe.set_ip_adapter_scale(scales)
        return pipe
