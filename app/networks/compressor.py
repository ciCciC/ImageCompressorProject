import numpy as np
from diffusers import AutoencoderTiny
from torchvision.transforms.functional import to_tensor
import torch
from PIL import Image
from typing import List, Tuple


class ImageCompressor:
    def __init__(self):
        self.d_type = None
        self.tiny_vae_model = None
        self.latent_dims = (4, 64, 64)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        if self.device in ['cuda', 'cpu']:
            self.d_type = torch.float16

        self.model_id = 'madebyollin/taesd'

    def load_model(self):
        self.tiny_vae_model = AutoencoderTiny.from_pretrained(self.model_id, torch_dtype=self.d_type).to(self.device)

    def compress(self, raw_images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Size]:
        tensor_block = torch.stack([to_tensor(img) for img in raw_images]).to(self.device)
        latent_space = self.tiny_vae_model.encoder(tensor_block)
        return latent_space, latent_space.shape

    def vector_ndarray(self, latent_tensor: torch.Tensor) -> np.ndarray:
        return latent_tensor.flatten().numpy(force=True)

    def latent_space_dimensionalize_tensor(self, latent_vector: np.ndarray) -> torch.Tensor:
        dim_shift = to_tensor(latent_vector.reshape(self.latent_dims))
        dim_moved = dim_shift.movedim(0, -1)
        return dim_moved.to(self.device)
