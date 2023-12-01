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
        self._np_d_type = np.float16
        self.model_id = 'madebyollin/taesd'

    def get_latent_dims(self):
        return self._latent_dims

    def get_latent_mu_size(self):
        return math.prod(self._latent_dims[:2])

    def load_model(self):
        self._model = (AutoencoderTiny
                       .from_pretrained(self.model_id, torch_dtype=self.d_type)
                       .to(self._device))

    def get_model(self):
        return self._model

    def compress(self, raw_images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Size]:
        tensor_block = (torch.stack([to_tensor(self.preprocess(img)) for img in raw_images])
                        .to(self._device, dtype=self.d_type))

        latent_space = self._model.encoder(tensor_block)
        return latent_space, latent_space.shape

    def preprocess(self, raw_image: Image.Image) -> Image.Image:
        return resize(raw_image, self._max_dim)

    def vector_ndarray(self, latent_tensor: torch.Tensor) -> np.ndarray:
        return latent_tensor.flatten().numpy(force=True).astype(self._np_d_type)

    def dimensionalize(self, latent_vector: List) -> torch.Tensor:
        reshaped = np.array(latent_vector, dtype=self._np_d_type).reshape(self._latent_dims)
        dim_shift = to_tensor(reshaped).movedim(0, -1).unsqueeze(0)
        return dim_shift

    def decompress(self, latent_vector: List) -> Image.Image:
        dim_shift_gpu = self.dimensionalize(latent_vector).to(self._device, dtype=self.d_type)
        reconstructed = self._model.decoder(dim_shift_gpu).clamp(0, 1)
        return to_pil_image(reconstructed[0])

    def decompress_batch(self, latent_space_block: List) -> torch.Tensor:
        dimensionalized_block = (torch.stack(
            [self.dimensionalize(latent_vector)[0] for latent_vector in latent_space_block])
                                 .to(self._device, dtype=self.d_type))

        reconstructed_block = self._model.decoder(dimensionalized_block).clamp(0, 1)
        return reconstructed_block

    def depict_latents(self, latent_vector: List) -> Image.Image:
        dim_shift_gpu = self.dimensionalize(latent_vector)
        latent_representation = Image.fromarray(
            dim_shift_gpu[0, :3].mul(0.25)
            .add(0.5)
            .clamp(0, 1)
            .mul(255).round().byte()
            .permute(1, 2, 0)
            .cpu().numpy())

        return latent_representation
