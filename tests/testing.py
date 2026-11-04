import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    # Seeding the *global* legacy RNG is deliberate: third-party code (PyG,
    # sklearn splitters) draws from it, which a local Generator cannot control.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
