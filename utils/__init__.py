import torch
import numpy as np
import random


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
