# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class BaseSheafConv(MessagePassing):
    """Shared base for all NSD and NSP sheaf convolution layers.

    Factors out the parameterization and utilities that are identical across
    every variant in both model families:

    * ``W1`` / ``W2`` -- bilateral stalk transforms (left d*d, right f*f).
    * ``sigma`` -- activation function (Tanh).
    * ``reset_parameters()`` -- Xavier init for W1, W2, and any map_generator.
    * ``_apply_stalk_transform(x)`` -- computes ``W1 @ x @ W2``.
    * ``_apply_norm(...)`` -- abstract; each concrete subclass delegates to the
      appropriate ``apply_*_norm`` function from ``sheaf_mpnn.utils``.

    Concrete subclasses must implement:
        ``get_map_products(x_feat, edge_index) -> (self_map, cross_map)``
        ``_apply_norm(self_map, cross_map, edge_index, num_nodes)``
        ``forward(x_feat, x_stalk, edge_index) -> updated stalk``
        ``message(...)``
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels  # 'f' (feature dimension per stalk entry)
        self.context_dim = (
            context_dim if context_dim is not None else (stalk_dim * in_channels)
        )
        self.add_self_loops = add_self_loops

        self.W1 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2 = nn.Parameter(torch.empty(in_channels, in_channels))  # [f, f]
        self.sigma = nn.Tanh()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        # Covers both the singular map_generator (NSD) and the
        # map_generators ModuleList (SheafAttnConv).
        generators: list[nn.Sequential] = []
        if hasattr(self, "map_generator"):
            gen = self.map_generator
            if isinstance(gen, nn.Sequential):
                generators.append(gen)
        if hasattr(self, "map_generators"):
            map_gens = self.map_generators
            if isinstance(map_gens, nn.ModuleList):
                for gen in map_gens:
                    if isinstance(gen, nn.Sequential):
                        generators.append(gen)
        for gen in generators:
            for m in gen:
                if isinstance(m, nn.Linear):
                    # gain=0.01: warm-start near-zero so the Laplacian is off at init.
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _apply_stalk_transform(self, x):
        """Applies bilateral stalk transform: W1 @ x @ W2."""
        return torch.matmul(torch.matmul(self.W1, x), self.W2)

    @abstractmethod
    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Normalizes restriction-map products by the sheaf degree matrix D^{-1/2}.

        Each concrete subclass calls the matching ``apply_*_norm`` utility from
        ``sheaf_mpnn.utils``:
            Diagonal   → apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)
            Orthogonal → apply_orthogonal_norm(cross_map, edge_index, num_nodes)
            Low-rank   → apply_low_rank_norm(self_map, cross_map, ..., stalk_dim, ...)
            General    → apply_general_norm(self_map, cross_map, ..., stalk_dim, ...)

        Returns:
            (norm_self, norm_cross): Normalized products ready for message().
        """
        raise NotImplementedError("Subclasses must implement _apply_norm.")

    def message(  # ty: ignore[invalid-method-override]
        self, z_dst, z_src, self_map, cross_map
    ):
        """Builds per-edge sheaf Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Normalized F_dst^T F_dst per edge [E, d, d].
            cross_map: Normalized F_dst^T F_src per edge [E, d, d].

        Returns:
            torch.Tensor: Per-edge messages [E, d, f].
        """
        return torch.matmul(self_map, z_dst) - torch.matmul(cross_map, z_src)


__all__ = ["BaseSheafConv"]
