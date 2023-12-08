import numpy as np
from diffusers import AutoencoderTiny
from torchvision.transforms.functional import to_tensor, resize, to_pil_image
import torch
from PIL import Image
from typing import List, Tuple
import math
from app.models.base_model import BaseModel


class ImageCompressor(BaseModel):
    def __init__(self):
        super().__init__()

        self._latent_dims = (4, 64, 64)
        self._max_dim = 512
        self._np_d_type = np.uint8
        self.model_id = 'madebyollin/taesd'

    def get_latent_dims(self):
        return self._latent_dims

    def get_latent_mu_size(self):
        return math.prod(self._latent_dims[:2])

    def load_model(self):
        self._model = (AutoencoderTiny
                       .from_pretrained(self.model_id,
                                        torch_dtype=self.d_type,
                                        use_safetensors=True)
                       .to(self._device))

    def get_model(self):
        return self._model

    @torch.no_grad()
    def compress(self, raw_images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Size]:
        tensor_block = (torch.stack([to_tensor(self.preprocess(img)) for img in raw_images])
                        .to(self._device, dtype=self.d_type))

        latent_space = self._model.encoder(tensor_block)
        return latent_space, latent_space.shape

    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """float16 1,4,64^2 <-> 1,4,64^2 to uint8"""
        scaled_latents = self._model.scale_latents(latents).mul_(255).round_().byte()
        return scaled_latents

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """float16 1,4,64^2 <-> 1,4,64^2 to float16"""
        unscaled_latents = self._model.unscale_latents(latents)
        return unscaled_latents

    def preprocess(self, raw_image: Image.Image) -> Image.Image:
        return resize(raw_image, self._max_dim)

    def vector_ndarray(self, latent_tensor: torch.Tensor) -> np.ndarray:
        return latent_tensor.flatten().numpy(force=True).astype(self._np_d_type)

    def dimensionalize(self, latent_vector: List) -> torch.Tensor:
        """uint8 to float16"""
        reshaped = np.array(latent_vector, dtype=self._np_d_type).reshape(self._latent_dims)
        dim_shift = to_tensor(reshaped).movedim(0, -1).unsqueeze(0)
        return dim_shift

    @torch.no_grad()
    def decompress(self, latent_vector: List) -> Image.Image:
        dim_shift_gpu = self.dimensionalize(latent_vector).to(self._device, dtype=self.d_type)
        unscaled_latents = self._model.unscale_latents(dim_shift_gpu)
        reconstructed = self._model.decoder(unscaled_latents).clamp(0, 1)
        return to_pil_image(reconstructed[0])

    @torch.no_grad()
    def decompress_by_image(self, image: Image.Image) -> Image.Image:
        tensor = to_tensor(image).unsqueeze(0).to(self._device, dtype=self.d_type)
        unscaled_latents = self._model.unscale_latents(tensor)
        reconstructed = self._model.decoder(unscaled_latents).clamp(0, 1)
        return to_pil_image(reconstructed[0])

    @torch.no_grad()
    def decompress_batch(self, latent_space_block: List) -> torch.Tensor:
        dimensionalized_block = (torch.stack(
            [self.dimensionalize(latent_vector)[0] for latent_vector in latent_space_block])
                                 .to(self._device, dtype=self.d_type))

        unscaled_block = self._model.unscale_latents(dimensionalized_block)

        reconstructed_block = self._model.decoder(unscaled_block).clamp(0, 1)
        return reconstructed_block

    @torch.no_grad()
    def decompress_batch_by_image(self, images: List[Image.Image]) -> torch.Tensor:
        tensor_block = (torch.stack([to_tensor(image) for image in images])
                        .to(self._device, dtype=self.d_type))

        unscaled_block = self._model.unscale_latents(tensor_block)

        reconstructed_block = self._model.decoder(unscaled_block).clamp(0, 1)
        return reconstructed_block

    def depict_latents(self, latent_vector: List) -> Image.Image:
        dim_shift_gpu = self.dimensionalize(latent_vector).to(dtype=self.d_type)
        return to_pil_image(dim_shift_gpu[0])
