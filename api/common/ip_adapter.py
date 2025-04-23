from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image

from common.exeptions import IPAdapterConfigError
from image.schemas import IpAdapterModel, IpAdapterModelConfig, ModelConfig
from utils.logger import logger
from utils.utils import load_image_if_exists

processor = IPAdapterMaskProcessor()


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
        self.image_path = data.image_path
        self.mask_path = data.mask_path
        self.scale = data.scale
        self.scale_layers = data.scale_layers
        self.image = load_image_if_exists(self.image_path)

        if not self.image:
            raise IPAdapterConfigError(f"Could not load IP-Adapter image from {self.image_path}")

        if self.scale < 0.01:
            raise IPAdapterConfigError("IP-Adapter scale must be >= 0.01")

        # we allways need a full mask if some use a mask
        self.mask_image = Image.new("L", (width, height), 255)  # Create 512x512 white image in L mode
        tmp_mask_image = load_image_if_exists(self.mask_path)
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
