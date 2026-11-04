# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Heterogeneous variational sheaf learners for Bayesian Sheaf NNs.

The BSNN restriction maps are latent random variables whose variational
distribution is parameterized per incident node-edge pair from the endpoint
features (Gillespie et al., 2025). The het learners widen that
parameterization with the standard het context sources:

* ``use_features``  -- the concatenated endpoint features ``[x_src, x_dst]``.
* ``use_node_types`` -- one-hot node types of both endpoints.
* ``use_edge_types`` -- the one-hot edge type.

The sampled maps and their KL divergences are unchanged -- only the input of
the mean/concentration heads grows, so the type information shapes the
posterior over sheaves rather than the diffusion itself.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.bsnn.variational import (
    DiagonalVariationalSheafLearner,
    GeneralVariationalSheafLearner,
    OrthogonalVariationalSheafLearner,
)


class HetVariationalContextMixin:
    """Widens the variational heads of a BSNN sheaf learner with het context.

    Hosts call ``_init_het_context()`` after the parent constructor: it
    validates the flags, recreates ``lin_mean`` / ``lin_scale`` with the het
    input width, and ``forward`` then builds the per-incident-pair input
    ``[x_src, x_dst, nt_src, nt_dst, et]`` from the enabled sources.
    """

    d: int
    context_dim: int
    mean_size: int
    scale_size: int
    lin_mean: nn.Linear
    lin_scale: nn.Linear
    num_node_types: int
    num_edge_types: int
    use_features: bool
    use_node_types: bool
    use_edge_types: bool

    def _init_het_context(
        self,
        num_node_types: int,
        num_edge_types: int,
        use_features: bool,
        use_node_types: bool,
        use_edge_types: bool,
    ) -> None:
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_features = use_features
        self.use_node_types = use_node_types
        self.use_edge_types = use_edge_types

        if use_node_types and num_node_types <= 0:
            raise ValueError(
                f"num_node_types must be > 0 when use_node_types=True, "
                f"got num_node_types={num_node_types}"
            )
        if use_edge_types and num_edge_types <= 0:
            raise ValueError(
                f"num_edge_types must be > 0 when use_edge_types=True, "
                f"got num_edge_types={num_edge_types}"
            )

        in_dim = 0
        if use_features:
            in_dim += 2 * self.context_dim
        if use_node_types:
            in_dim += 2 * num_node_types
        if use_edge_types:
            in_dim += num_edge_types
        if in_dim <= 0:
            raise ValueError(
                "Invalid het-context configuration: computed learner input "
                "width is 0. Enable at least one of use_features, "
                "use_node_types, or use_edge_types."
            )

        # Replace the heads sized by the parent constructor with het-wide ones
        # (bias-free, following the reference learner).
        self.lin_mean = nn.Linear(in_dim, self.mean_size, bias=False)
        self.lin_scale = nn.Linear(in_dim, self.scale_size, bias=False)

    def _het_pair_input(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenated het context per directed edge; row ``i`` parameterizes
        the map of the incident pair ``src_i`` ⊴ ``e_i``.
        """
        src_idx, dst_idx = edge_index
        parts: list[torch.Tensor] = []

        if self.use_features:
            parts.extend([x_feat[src_idx], x_feat[dst_idx]])

        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
            else:
                nt_oh = torch.zeros(
                    x_feat.size(0),
                    self.num_node_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts.extend([nt_oh[src_idx], nt_oh[dst_idx]])

        if self.use_edge_types:
            if edge_types is not None:
                et_oh = F.one_hot(edge_types, self.num_edge_types).to(x_feat.dtype)
            else:
                et_oh = torch.zeros(
                    edge_index.size(1),
                    self.num_edge_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts.append(et_oh)

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute variational parameters for every incident node-edge pair.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            edge_index: Graph connectivity ``[2, E]`` (both orientations of
                every undirected edge).
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.

        Returns:
            ``(mean_params, scale_params)`` -- raw head outputs, transformed
            by the variant in ``sample`` / ``kl``.
        """
        inp = self._het_pair_input(x_feat, edge_index, node_types, edge_types)
        return self.lin_mean(inp), self.lin_scale(inp)


class HetDiagonalVariationalSheafLearner(
    HetVariationalContextMixin, DiagonalVariationalSheafLearner
):
    """Gaussian posterior over diagonal maps with het-context heads."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
    ):
        super().__init__(stalk_dim, in_channels, context_dim, sparse=False)
        self._init_het_context(
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
        )


class HetGeneralVariationalSheafLearner(
    HetVariationalContextMixin, GeneralVariationalSheafLearner
):
    """Gaussian posterior over general linear maps with het-context heads."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
    ):
        super().__init__(stalk_dim, in_channels, context_dim, sparse=False)
        self._init_het_context(
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
        )


class HetOrthogonalVariationalSheafLearner(
    HetVariationalContextMixin, OrthogonalVariationalSheafLearner
):
    """Cayley-distribution posterior over SO(d) maps with het-context heads."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            context_dim,
            sparse=False,
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
        )
        self._init_het_context(
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
        )


# Concrete het learner types; used to annotate HetBSNNModel.sheaf_learner so
# type checkers resolve the het forward/sample/kl signatures.
HetVariationalSheafLearner = (
    HetDiagonalVariationalSheafLearner
    | HetGeneralVariationalSheafLearner
    | HetOrthogonalVariationalSheafLearner
)


__all__ = [
    "HetVariationalContextMixin",
    "HetDiagonalVariationalSheafLearner",
    "HetGeneralVariationalSheafLearner",
    "HetOrthogonalVariationalSheafLearner",
    "HetVariationalSheafLearner",
]
