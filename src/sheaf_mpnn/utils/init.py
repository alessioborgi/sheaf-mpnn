# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Shared initialization for restriction-map generators."""

import torch
from torch import nn


def init_map_generator_(
    map_generator: nn.Sequential, stalk_dim: int, map_variant: str
) -> None:
    """Init a map generator so the cross map F_u^T F_v starts near identity.

    The transported quantity is the cross map F_u^T F_v, a product of two
    generated maps, so its order in the generator output phi differs per variant.
    A blanket near-zero init (phi ~ 0) gives identity transport only for the
    orthogonal variant (cayley(0) = I); for the others it makes the cross map
    vanish as O(phi^2) (diagonal/general) or O(phi^4) (low_rank), killing signal
    and gradient at init. So we set the final-layer bias to the identity per
    variant (diagonal -> 1, general -> vec(I_d)) with a small weight; low_rank
    cannot reach I for rank < d, so it uses a standard-scale (non-degenerate)
    init instead.

    Shared by SheafGIN/SheafDeepSet and SheafSetTransformer, which all build the
    same [Linear, ReLU, Linear] generator. Operates in place on map_generator
    only; other MLPs (mu, psi) keep their own init. map_variant must be one of
    diagonal/general/orthogonal/low_rank.
    """
    hidden, final = map_generator[0], map_generator[-1]
    assert isinstance(hidden, nn.Linear) and isinstance(final, nn.Linear)
    nn.init.xavier_uniform_(hidden.weight, gain=1.0)
    if hidden.bias is not None:
        nn.init.constant_(hidden.bias, 0.0)

    d = stalk_dim
    if map_variant == "low_rank":
        nn.init.xavier_uniform_(final.weight, gain=1.0)
        if final.bias is not None:
            nn.init.constant_(final.bias, 0.0)
        return

    # orthogonal/diagonal/general: small final weights so phi ~ final bias.
    nn.init.xavier_uniform_(final.weight, gain=0.01)
    if final.bias is None:
        return
    if map_variant == "diagonal":
        nn.init.constant_(final.bias, 1.0)  # diag(1) = I
    elif map_variant == "general":
        # final bias = vec(I_d), row-major to match phi.view(-1, d, d).
        eye = torch.eye(d, device=final.bias.device, dtype=final.bias.dtype)
        with torch.no_grad():
            final.bias.copy_(eye.reshape(-1))
    else:  # orthogonal: cayley(0) = I already
        nn.init.zeros_(final.bias)
