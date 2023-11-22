from diffusers import DiffusionPipeline, LCMScheduler, StableDiffusionUpscalePipeline, UniPCMultistepScheduler, \
    AutoencoderTiny
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, center_crop, resize, to_tensor
import torch
from typing import List
from base_model import BaseModel
from PIL import Image


class ImageGenerator(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_id = 'Lykon/dreamshaper-7'

    def load_model(self):
        self._model = DiffusionPipeline.from_pretrained(self.model_id,
                                                        torch_dtype=self._d_type,
                                                        use_safetensors=True)

    def optimize(self):
        self._model.scheduler = LCMScheduler.from_config(self._model.scheduler.config)
        self._model.enable_attention_slicing()
        self._model.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self._model.fuse_lora()

    def inference(self, prompt: str):
        pass

    def multi_inference(self, prompts: List[str]):
        pass
