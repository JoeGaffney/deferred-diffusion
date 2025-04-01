from utils.logger import logger
from utils.utils import load_image_if_exists


class IpAdapter:
    def __init__(self, model, image_path, strength, width, height):
        self.model = model
        self.image_path = image_path
        self.strength = strength
        self.adapter_model = "h94/IP-Adapter"
        self.subfolder = "models"
        self.weight_name = "ip-adapter_sd15.bin"
        self.enabled = False

        if "stable-diffusion-v1-5" in model:
            self.adapter_model = "h94/IP-Adapter"
            self.subfolder = "models"
            self.weight_name = "ip-adapter_sd15.bin"
            self.enabled = True
        elif "stable-diffusion-xl" in model:
            self.adapter_model = "h94/IP-Adapter"
            self.subfolder = "sdxl_models"
            self.weight_name = "ip-adapter_sdxl.bin"
            self.enabled = True

        self.image = load_image_if_exists(self.image_path)
        if self.image:
            print("skip resize?")
            # self.image = self.image.resize([width, height])
        else:
            self.enabled = False

    def load_adapter(self, pipe):
        # NOTE possibly we need to unload if we are caching pipelines
        # if hasattr(pipe, "unload_ip_adapter"):
        #     logger.warning(f"Unloading IP Adapter {self.adapter_model}")
        #     pipe.unload_ip_adapter()

        if self.enabled:

            if hasattr(pipe, "load_ip_adapter"):
                # Store current device
                device = pipe.device
                logger.warning(
                    f"Loading IP Adapter {device} - {self.adapter_model} {self.subfolder} {self.weight_name} {self.image_path}"
                )

                # Load IP adapter
                pipe.load_ip_adapter(
                    self.adapter_model,
                    subfolder=self.subfolder,
                    weight_name=self.weight_name,
                )

                pipe.set_ip_adapter_scale(self.strength)

                # Move to CUDA - the pipeline's CPU offload will handle subsequent device management?
                pipe.to("cuda")
            else:
                logger.warning("IP Adapter not supported for this model")
                self.enabled = False
        return pipe
