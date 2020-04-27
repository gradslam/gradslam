import torch
if torch.cuda.is_available():
    from .chamfer import *
