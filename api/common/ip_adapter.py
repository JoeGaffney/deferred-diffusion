from diffusers.image_processor import IPAdapterMaskProcessor
from PIL import Image

from image.schemas import IpAdapterModel
from utils.logger import logger
from utils.utils import load_image_if_exists

processor = IPAdapterMaskProcessor()


class IpAdapter:
    def __init__(self, data: IpAdapterModel, width, height):
        self.model = data.model
        self.image_path = data.image_path
        self.mask_path = data.mask_path
        self.scale = data.scale
        self.scale_layers = data.scale_layers
        self.subfolder = data.subfolder
        self.weight_name = data.weight_name
        self.image = load_image_if_exists(self.image_path)
        self.mask_image = load_image_if_exists(self.mask_path)
        self.image_encoder = data.image_encoder
        self.image_encoder_subfolder = "models/image_encoder"
        self.enabled = False
        self.solid_mask = Image.new("L", (width, height), 255)  # Create 512x512 white image in L mode

        if self.image and self.scale > 0.01:
            # NOTE should we resize the image to the input size?
            # self.image = self.image.resize([512, 512])
            # self.image = self.image.resize([width, height])
            if self.model is not None and self.subfolder is not None and self.weight_name is not None:
                self.enabled = True

            if self.mask_image is not None:
                self.mask_image = self.mask_image.resize([width, height])
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
        tmp_mask = self.solid_mask
        if self.mask_image:
            tmp_mask = self.mask_image

        tmp_mask.save(f"{self.mask_path}_{self.scale}_resized.png")
        return processor.preprocess(tmp_mask)
