from image.schemas import IpAdapterModel
from utils.logger import logger
from utils.utils import load_image_if_exists


class IpAdapter:
    def __init__(self, data: IpAdapterModel, width, height):
        self.model = data.model
        self.image_path = data.image_path
        self.scale = data.scale
        self.subfolder = data.subfolder
        self.weight_name = data.weight_name
        self.image = load_image_if_exists(self.image_path)
        self.enabled = False

        if self.model is None or self.image_path is None:
            self.enabled = False

        if self.image:
            # NOTE should we resize the image to the input size?
            # self.image = self.image.resize([width, height])
            if self.model is not None and self.subfolder is not None and self.weight_name is not None:
                self.enabled = True
