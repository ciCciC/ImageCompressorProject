import numpy as np
from diffusers import AutoencoderTiny
from torchvision.transforms.functional import to_tensor, resize
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
        self.model_id = 'madebyollin/taesd'

    def get_latent_dims(self):
        return self._latent_dims

    def get_latent_flat_size(self):
        return math.prod(self._latent_dims)

    def load_model(self):
        self._model = AutoencoderTiny.from_pretrained(self.model_id, torch_dtype=self._d_type).to(self._device)

    def compress(self, raw_images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Size]:
        tensor_block = torch.stack([to_tensor(self.preprocess(img)) for img in raw_images]).to(self._device)
        latent_space = self._model.encoder(tensor_block)
        return latent_space, latent_space.shape

    def preprocess(self, raw_image: Image.Image) -> Image.Image:
        return resize(raw_image, self._max_dim)

    def vector_ndarray(self, latent_tensor: torch.Tensor) -> np.ndarray:
        return latent_tensor.flatten().numpy(force=True)

    def latent_space_dimensionalize_tensor(self, latent_vector: np.ndarray) -> torch.Tensor:
        dim_shift = to_tensor(latent_vector.reshape(self._latent_dims))
        dim_moved = dim_shift.movedim(0, -1)
        return dim_moved.to(self._device)
