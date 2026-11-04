# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Identity-sheaf diffusion (Hernandez Caralt et al., 2024)."""

from .id_sheaf_layers import IdentitySheafConv
from .id_sheaf_model import IdentitySheafModel

__all__ = [
    "IdentitySheafConv",
    "IdentitySheafModel",
]
