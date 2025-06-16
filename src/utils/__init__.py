import random

import numpy as np
import torch

from .arguments import get_args, get_exp_name


def set_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(seed)
