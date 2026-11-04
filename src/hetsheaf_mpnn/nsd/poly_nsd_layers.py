# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from hetsheaf_mpnn.nsd.nsd_layers import (
    DiagonalHetNSDConv,
    GeneralHetNSDConv,
    LowRankHetNSDConv,
    OrthogonalHetNSDConv,
)

if TYPE_CHECKING:
    from typing import Protocol

    class _HetNSDConvProto(Protocol):
        stalk_dim: int
        sigma: Any
        add_self_loops: bool

        def reset_parameters(self) -> None: ...
        def _apply_stalk_transform(self, x: torch.Tensor) -> torch.Tensor: ...
        def get_map_products(
            self,
            x_feat: torch.Tensor,
            edge_index: torch.Tensor,
            node_types: torch.Tensor | None,
            edge_types: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, torch.Tensor]: ...
        def _apply_norm(
            self,
            self_map: torch.Tensor | None,
            cross_map: torch.Tensor,
            edge_index: torch.Tensor,
            num_nodes: int,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...
        def propagate(
            self, edge_index: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor: ...

    _PolyHetNSDBase = _HetNSDConvProto
else:
    _PolyHetNSDBase = object


class BasePolyHetNSDConv(_PolyHetNSDBase):
    """Mixin: het-NSD forward replaced by a degree-K Chebyshev polynomial filter.

    Usage: place before the het-NSD conv in the MRO, e.g.
        class DiagonalPolyHetNSDConv(BasePolyHetNSDConv, DiagonalHetNSDConv): ...

    Identical in structure to BasePolyNSDConv but the forward signature includes
    node_types and edge_types, which are forwarded to get_map_products so that
    the restriction-map MLP can condition on graph heterogeneity.

    Output:
        poly  = sum_k softmax(theta_raw)_k * T_k(L_tilde)  (Chebyshev recurrence)
        gate  = 1 + tanh(eps)
        x_out = gate * x_stalk - sigma(poly + hp_alpha * (z - Delta_F(z) / lambda_max))

    where L_tilde = (2/lambda_max)*Delta_F - I rescales spectrum
    [0, lambda_max] -> [-1, 1].
    """

    def __init__(
        self,
        *args,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = degree
        self.lambda_max = lambda_max
        self.theta_raw = nn.Parameter(torch.zeros(degree + 1))
        self.hp_alpha = nn.Parameter(torch.tensor(float(hp_alpha_init)))
        self.eps = nn.Parameter(torch.zeros(self.stalk_dim))

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "theta_raw"):
            nn.init.zeros_(self.theta_raw)
        if hasattr(self, "hp_alpha"):
            nn.init.zeros_(self.hp_alpha)
        if hasattr(self, "eps"):
            nn.init.zeros_(self.eps)

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one het PolyNSD diffusion step.

        Args:
            x_feat:     [N, context_dim]            Flattened stalk (context for maps).
            x_stalk:    [N, stalk_dim, in_channels]  Structured stalk to update.
            edge_index: [2, E]                       Graph connectivity.
            node_types: [N] LongTensor or None       Node-type indices.
            edge_types: [E] LongTensor or None       Edge-type indices.

        Returns:
            Updated stalk [N, stalk_dim, in_channels].
        """
        z = self._apply_stalk_transform(x_stalk)
        num_nodes = x_stalk.size(0)

        # Mirror BaseHetNSDConv.forward: augment with self-loops before degree
        # normalisation.  edge_types must be padded for the new self-loop edges
        # (type 0) so type-conditioned map generation stays aligned with
        # edge_index when edge_types is provided.
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        src_idx, dst_idx = edge_index

        # Compute restriction-map products once; reused for all recurrence steps.
        self_map, cross_map = self.get_map_products(
            x_feat, edge_index, node_types, edge_types
        )
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        theta = torch.softmax(self.theta_raw, dim=0)  # [K+1]

        def _L(h: torch.Tensor) -> torch.Tensor:
            return self.propagate(
                edge_index,
                z_dst=h[dst_idx],
                z_src=h[src_idx],
                self_map=norm_self,
                cross_map=norm_cross,
                size=(num_nodes, num_nodes),
            )

        def _L_tilde(h: torch.Tensor) -> torch.Tensor:
            return (2.0 / self.lambda_max) * _L(h) - h

        Lz = _L(z)  # raw Delta_F(z); reused for HP skip and L_tilde(z) derivation
        L_tilde_z = (2.0 / self.lambda_max) * Lz - z  # spectrum [-1, 1]

        T_prev, T_curr = z, L_tilde_z
        result = theta[0] * T_prev + theta[1] * T_curr

        for k in range(2, self.degree + 1):
            T_next = 2.0 * _L_tilde(T_curr) - T_prev
            result = result + theta[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        result = result + self.hp_alpha * (z - Lz / self.lambda_max)

        gate = (1.0 + torch.tanh(self.eps)).view(1, -1, 1)
        return gate * x_stalk - self.sigma(result)


# ---------------------------------------------------------------------------
# Concrete polynomial het-NSD variants
# ---------------------------------------------------------------------------


class DiagonalPolyHetNSDConv(BasePolyHetNSDConv, DiagonalHetNSDConv):
    """Polynomial het-NSD with diagonal restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
        )


class GeneralPolyHetNSDConv(BasePolyHetNSDConv, GeneralHetNSDConv):
    """Polynomial het-NSD with general (full dxd) restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            use_attention=use_attention,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
        )


class OrthogonalPolyHetNSDConv(BasePolyHetNSDConv, OrthogonalHetNSDConv):
    """Polynomial het-NSD with orthogonal restriction maps.

    For orthogonal maps F^T F = I, so self_map = I.  The forward is overridden
    in OrthogonalHetNSDConv to skip the self_map term; BasePolyHetNSDConv's
    forward calls _apply_norm(None, cross_map, ...) which routes to
    apply_orthogonal_norm, matching the first-order behaviour.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        clamp_val: float = 10.0,
        use_attention: bool = False,
        orth_strategy: str = "cayley",
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            clamp_val=clamp_val,
            use_attention=use_attention,
            orth_strategy=orth_strategy,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
        )


class LowRankPolyHetNSDConv(BasePolyHetNSDConv, LowRankHetNSDConv):
    """Polynomial het-NSD with low-rank restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        rank: int = 1,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            rank=rank,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
        )


__all__ = [
    "BasePolyHetNSDConv",
    "DiagonalPolyHetNSDConv",
    "GeneralPolyHetNSDConv",
    "OrthogonalPolyHetNSDConv",
    "LowRankPolyHetNSDConv",
]
