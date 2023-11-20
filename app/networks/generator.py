from diffusers import DiffusionPipeline, LCMScheduler, StableDiffusionUpscalePipeline, UniPCMultistepScheduler, \
    AutoencoderTiny
from torchvision.transforms.functional import pil_to_tensor, to_pil_image, center_crop, resize, to_tensor
import torch
from PIL import Image


class ImageGenerator:

    def __init__(self, model_id: str = 'Lykon/dreamshaper-7'):
        self.d_type = None
        self.txt_2_img_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        if self.device in ['cuda', 'cpu']:
            self.d_type = torch.float16

        self.model_id = model_id

    def load_model(self):
        self.txt_2_img_model = DiffusionPipeline.from_pretrained(self.model_id,
                                                                 torch_dtype=self.d_type,
                                                                 use_safetensors=True)

    def optimize(self):
        self.txt_2_img_model.scheduler = LCMScheduler.from_config(self.txt_2_img_model.scheduler.config)
        self.txt_2_img_model.enable_attention_slicing()
        self.txt_2_img_model.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.txt_2_img_model.fuse_lora()

    def inference(self, prompt):
        pass

    def multi_inference(self, prompts):
        pass
