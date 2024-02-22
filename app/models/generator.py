import torch
from diffusers import DiffusionPipeline, LCMScheduler
from typing import List, Tuple
from app.models.base_model import BaseModel
from PIL import Image
from app.core.settings import MODEL_WEIGHTS_DIR


class ImageGenerator(BaseModel):

    def __init__(self):
        super().__init__()
        self.model_id = f'{MODEL_WEIGHTS_DIR}/dreamshaper-8'

    def load_model(self):
        self._model = DiffusionPipeline.from_pretrained(self.model_id,
                                                        torch_dtype=self.d_type,
                                                        use_safetensors=True)

        self._optimize()

    def _optimize(self):
        # self._model.scheduler = LCMScheduler.from_config(self._model.scheduler.config)
        # self._model.enable_attention_slicing()

        self._model = self._model.to(self.device)

        # self._model.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        # self._model.fuse_lora()

    @torch.inference_mode()
    def inference(self, prompt: str) -> Tuple[List[Image.Image], bool]:
        results = self._model(
            prompt=prompt,
            num_inference_steps=4
        )

        is_nsfw: bool = results.nsfw_content_detected[0]
        image: List[Image.Image] = results.images

        return image, is_nsfw

    @torch.inference_mode()
    def multi_inference(self, prompts: List[str]) -> Tuple[List[Image.Image], List[bool]]:
        images: List[Image.Image] = []
        nsfws: List[bool] = []

        for prompt in prompts:
            result = self._model(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
            )
            images.append(result.images[0])
            nsfws.append(result.nsfw_content_detected[0])

        return images, nsfws
