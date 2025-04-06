from image.schemas import IpAdapterModel
from utils.logger import logger
from utils.utils import load_image_if_exists


class IpAdapter:
    def __init__(self, data: IpAdapterModel, width, height):
        self.model = data.model
        self.image_path = data.image_path
        self.scale = data.scale
        self.scale_layers = data.scale_layers
        self.subfolder = data.subfolder
        self.weight_name = data.weight_name
        self.image = load_image_if_exists(self.image_path)
        self.image_encoder = data.image_encoder
        self.image_encoder_subfolder = "models/image_encoder"
        self.enabled = False

        if self.model is None or self.image_path is None:
            self.enabled = False

        if self.image:
            # NOTE should we resize the image to the input size?
            # self.image = self.image.resize([512, 512])
            # self.image = self.image.resize([width, height])
            if self.model is not None and self.subfolder is not None and self.weight_name is not None:
                self.enabled = True

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
