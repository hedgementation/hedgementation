import os
import torch
from dotenv import load_dotenv
import numpy as np
import random


def set_random_seeds():
        """Set random seeds for reproducibility."""
        try:
            seed = int(os.environ["RANDOM_SEED"])
        except (KeyError, ValueError):
            seed = 15

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
