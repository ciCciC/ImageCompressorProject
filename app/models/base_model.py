import torch


class BaseModel:
    def __init__(self):
        self._model = None
        self.model_id = None
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.d_type = torch.float16
