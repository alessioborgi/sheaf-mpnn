# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch

from hetsheaf_mpnn.nsd.nsd_layers import BaseHetNSDConv
from sheaf_mpnn.utils import apply_diagonal_norm


class HetIdentitySheafConv(BaseHetNSDConv):
    """Heterogeneous Identity Sheaf Network (ISN) layer.

    Restriction maps are fixed to the identity, following Hernandez Caralt
    et al. (2024): the sheaf Laplacian reduces to the (deg+1)-augmented graph
    Laplacian per stalk channel and no map generator exists. ``node_types``
    and ``edge_types`` are accepted for het-pipeline parity but cannot
    influence the maps by construction -- ISN removes all learning from the
    sheaf, so the diffusion is deliberately type-blind.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            add_eps=add_eps,
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        """Identity maps: F^T F = I, so both products are all-ones diagonals."""
        num_edges = edge_index.size(1)
        ones = torch.ones(
            (num_edges, self.stalk_dim), dtype=x_feat.dtype, device=x_feat.device
        )
        return ones, ones

    def message(self, z_dst, z_src, self_map, cross_map):
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src


__all__ = ["HetIdentitySheafConv"]
