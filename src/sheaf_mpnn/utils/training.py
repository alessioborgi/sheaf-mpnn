# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import logging
import random

import numpy as np
import torch

_log = logging.getLogger(__name__)


def setup_torch(precision: str = "high", seed: int = 42) -> None:
    """Sets precision for float32 matrix multiplications and random seeds.

    Configures PyTorch and NumPy for reproducibility and performance.
    """
    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # Enable TF32 for matrix multiplications
        )
        _log.info("Using GPU: %s", torch.cuda.get_device_name(0))
        _ver = torch.version  # ty: ignore[possibly-missing-submodule]
        cuda_ver = getattr(_ver, "cuda", "N/A")
        _log.info("CUDA version: %s", cuda_ver)
        _log.info("cuDNN version: %s", torch.backends.cudnn.version())
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _log.info(
                "Using Apple Silicon GPU (MPS), "
                "falling back to CPU for reproducibility."
            )
            setattr(torch.backends.mps, "is_available", lambda: False)  # noqa: B010
        else:
            _log.info("Using CPU")

    _log.info("Float32 matmul precision set to: %s", precision)
    _log.info("Random seed set to: %d", seed)
    return None
