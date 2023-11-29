import numpy as np

from app.models.compressor import ImageCompressor
from app.models.generator import ImageGenerator
from typing import Tuple
import torch
import asyncio


class NeuralService:

    def __init__(self):
        self.compressor = ImageCompressor()
        self.generator = ImageGenerator()
        self.load_models()

    def load_models(self):
        self.compressor.load_model()
        self.generator.load_model()

    def prompt_inference(self, prompt: str) -> Tuple[np.ndarray, torch.Size, bool]:
        image, is_nsfw = self.generator.inference(prompt)
        latents, shape = self.compressor.compress(image)
        vectorized = self.compressor.vector_ndarray(latents[0])
        return vectorized, shape, is_nsfw

    async def prompt_inference_async(self, prompt: str) -> Tuple[np.ndarray, torch.Size, bool]:
        return await asyncio.to_thread(self.prompt_inference, prompt)
