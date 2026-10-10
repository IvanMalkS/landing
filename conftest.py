import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Sets deterministic seeds for random generators before every test."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
