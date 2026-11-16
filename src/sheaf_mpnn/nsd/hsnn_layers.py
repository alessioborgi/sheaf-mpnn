# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Restriction-map parameterizations for undirected Hypergraph Sheaf NNs."""

from typing import Literal

import torch

from sheaf_mpnn.hypergraph_sheaf_conv import BaseHypergraphSheafConv
from sheaf_mpnn.utils.orthogonal import (
    ORTH_STRATEGIES,
    build_orthogonal,
    orth_param_count,
)


class DiagonalHSNNConv(BaseHypergraphSheafConv):
    """HSNN with diagonal restriction maps, stored as [K, d] vectors.

    Arguments follow BaseHypergraphSheafConv. Each map is predicted from the
    concatenated node and mean-hyperedge contexts using Linear + Tanh.
    """

    diagonal_maps = True

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
    ) -> None:
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            dropout,
            residual,
            degree_regularization,
        )
        self.map_generator = self._build_map_generator(stalk_dim)
        self.reset_parameters()

    def parameterize_maps(self, context: torch.Tensor) -> torch.Tensor:
        """Predict bounded diagonal entries [K, d] from contexts [K, 2*C]."""
        return self.map_generator(context)


class GeneralHSNNConv(BaseHypergraphSheafConv):
    """HSNN with unrestricted d-by-d maps predicted by Linear + Tanh.

    Arguments follow BaseHypergraphSheafConv. Full node-degree blocks retain
    cross-channel interactions and gradients through their inverse square roots.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
    ) -> None:
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            dropout,
            residual,
            degree_regularization,
        )
        self.map_generator = self._build_map_generator(stalk_dim * stalk_dim)
        self.reset_parameters()

    def parameterize_maps(self, context: torch.Tensor) -> torch.Tensor:
        """Predict full restriction maps [K, d, d] from contexts [K, 2*C]."""
        return self.map_generator(context).reshape(
            context.size(0), self.stalk_dim, self.stalk_dim
        )


class OrthogonalHSNNConv(BaseHypergraphSheafConv):
    """HSNN with orthogonal restriction maps using the existing native tools.

    Args:
        stalk_dim: Positive stalk dimension d.
        in_channels: Feature channels f per stalk entry.
        hidden_dim: Positive hidden width, retained for the shared layer API.
        context_dim: Context width C, defaulting to d*f.
        dropout: Dropout probability on the propagation input.
        residual: Add the input before ELU.
        degree_regularization: Nonnegative node-degree regularization.
        orth_strategy: Cayley, FastH, compact FastH++, or Householder strategy.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
    ) -> None:
        if orth_strategy not in ORTH_STRATEGIES:
            raise ValueError(f"orth_strategy must be one of {ORTH_STRATEGIES}")
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            dropout,
            residual,
            degree_regularization,
        )
        self.orth_strategy = orth_strategy
        self.map_generator = self._build_map_generator(
            orth_param_count(orth_strategy, stalk_dim)
        )
        self.reset_parameters()

    def parameterize_maps(self, context: torch.Tensor) -> torch.Tensor:
        """Predict orthogonal maps [K, d, d] from contexts [K, 2*C]."""
        return build_orthogonal(
            self.map_generator(context), self.stalk_dim, self.orth_strategy
        )


class LowRankHSNNConv(BaseHypergraphSheafConv):
    """HSNN with pure low-rank maps F = A B.T, without a diagonal addition.

    Args:
        stalk_dim: Positive stalk dimension d.
        in_channels: Feature channels f per stalk entry.
        hidden_dim: Positive hidden width, retained for the shared layer API.
        context_dim: Context width C, defaulting to d*f.
        dropout: Dropout probability on the propagation input.
        residual: Add the input before ELU.
        degree_regularization: Nonnegative node-degree regularization.
        rank: Factor width r, with 1 <= r <= d. Degree blocks may be singular.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
        rank: int = 1,
    ) -> None:
        if not 1 <= rank <= stalk_dim:
            raise ValueError("rank must satisfy 1 <= rank <= stalk_dim")
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            dropout,
            residual,
            degree_regularization,
        )
        self.rank = rank
        self.map_generator = self._build_map_generator(2 * stalk_dim * rank)
        self.reset_parameters()

    def parameterize_maps(self, context: torch.Tensor) -> torch.Tensor:
        """Predict rank-at-most-r maps [K, d, d] from contexts [K, 2*C]."""
        factors = self.map_generator(context).reshape(
            context.size(0), self.stalk_dim, 2 * self.rank
        )
        left, right = factors.chunk(2, dim=-1)
        return left @ right.transpose(-1, -2)


__all__ = [
    "DiagonalHSNNConv",
    "GeneralHSNNConv",
    "OrthogonalHSNNConv",
    "LowRankHSNNConv",
]
