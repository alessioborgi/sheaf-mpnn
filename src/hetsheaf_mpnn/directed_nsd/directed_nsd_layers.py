# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod
from typing import Literal

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from hetsheaf_mpnn.base_conv import HetContextMixin
from sheaf_mpnn.base_conv import BaseDirectedSheafConv
from sheaf_mpnn.utils import (
    apply_directed_diagonal_norm,
    apply_directed_general_norm,
    apply_directed_low_rank_norm,
    apply_directed_orthogonal_norm,
    attention_cayley,
    cayley,
    householder,
)


class BaseHetDirectedNSDConv(HetContextMixin, BaseDirectedSheafConv):
    """Heterogeneous directed NSD convolution with two-direction message passing.

    Separately aggregates src→dst and dst→src directions with independent
    restriction maps, directional sheaf degree normalizations, and separate
    bilateral stalk transforms per direction (Dir-GNN approach), while the
    restriction-map MLPs consume the het context (features and node/edge-type
    one-hots) controlled by the standard het flags.

    Concrete subclasses must implement ``get_map_products()`` and
    ``_apply_norm()``.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        learn_alpha: bool = False,
    ):
        super().__init__(
            stalk_dim, in_channels, hidden_dim, context_dim, add_self_loops
        )
        self._init_het_context(
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=learn_alpha)

    def reset_parameters(self):
        # Extends BaseDirectedSheafConv (directional Ws plus the singular
        # map_generator) with the ensemble map_generators ModuleList.
        super().reset_parameters()
        map_gens = getattr(self, "map_generators", None)
        if isinstance(map_gens, nn.ModuleList):
            for gen in map_gens:
                for m in gen:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)

    def _map_pair(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates the (dst-perspective, src-perspective) raw map parameters."""
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        w_dst, w_src = raw.chunk(2, dim=0)
        return w_dst, w_src

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies het directed NSD with two-direction aggregation.

        Args:
            x_feat:     [N, context_dim]             Node context features.
            x_stalk:    [N, stalk_dim, in_channels]  Lifted node features.
            edge_index: [2, E]                       Directed connectivity.
            node_types: [N] LongTensor or None       Node-type indices.
            edge_types: [E] LongTensor or None       Edge-type indices.

        Returns:
            Updated stalk features [N, stalk_dim, in_channels].
        """
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        src_idx, dst_idx = edge_index

        dst_map, src_map, cross_map = self.get_map_products(
            x_feat, edge_index, node_types, edge_types
        )

        norm_fwd, norm_bwd = self._apply_norm(
            dst_map, src_map, cross_map, edge_index, num_nodes
        )

        z_fwd = self._apply_stalk_transform_fwd(x_stalk)
        z_bwd = self._apply_stalk_transform_bwd(x_stalk)

        agg_fwd = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_src=z_fwd[src_idx],
            F=norm_fwd,
            size=(num_nodes, num_nodes),
        )
        agg_bwd = self.propagate(  # ty: ignore[missing-argument]
            edge_index.flip(0),  # ty: ignore[invalid-argument-type]
            z_src=z_bwd[dst_idx],
            F=norm_bwd,
            size=(num_nodes, num_nodes),
        )

        blended = self.alpha * agg_fwd + (1 - self.alpha) * agg_bwd
        return self.sigma(blended)

    @abstractmethod
    def get_map_products(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Precompute (dst_map, src_map, cross_map) per edge."""

    @abstractmethod
    def _apply_norm(
        self,
        dst_map: torch.Tensor | None,
        src_map: torch.Tensor | None,
        cross_map: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise restriction-map products via directed sheaf degree matrices.

        Returns (norm_fwd, norm_bwd).
        """


class DiagonalHetDirectedNSDConv(BaseHetDirectedNSDConv):
    """Diagonal het directed NSD convolution."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        learn_alpha: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            learn_alpha,
        )
        self._make_map_generators(hidden_dim, stalk_dim)
        self.reset_parameters()

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        w_dst, w_src = self._map_pair(x_feat, edge_index, node_types, edge_types)
        return w_dst**2, w_src**2, w_dst * w_src

    def _apply_norm(self, dst_map, src_map, cross_map, edge_index, num_nodes):
        return apply_directed_diagonal_norm(
            dst_map, src_map, cross_map, edge_index, num_nodes
        )

    def message(self, z_src: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        return F[:, :, None] * z_src


class GeneralHetDirectedNSDConv(BaseHetDirectedNSDConv):
    """General het directed NSD convolution."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        learn_alpha: bool = False,
        use_attention: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            learn_alpha,
        )
        self.use_attention = use_attention
        self._make_map_generators(hidden_dim, stalk_dim * stalk_dim)
        self.reset_parameters()

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        phi_dst, phi_src = (
            w.view(-1, self.stalk_dim, self.stalk_dim)
            for w in self._map_pair(x_feat, edge_index, node_types, edge_types)
        )

        if self.use_attention:
            eye = torch.eye(
                self.stalk_dim, device=phi_dst.device, dtype=phi_dst.dtype
            ).unsqueeze(0)
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)

        dst_map = phi_dst.transpose(-2, -1) @ phi_dst  # [E, d, d]
        src_map = phi_src.transpose(-2, -1) @ phi_src  # [E, d, d]
        cross_map = phi_dst.transpose(-2, -1) @ phi_src  # [E, d, d]
        return dst_map, src_map, cross_map

    def _apply_norm(self, dst_map, src_map, cross_map, edge_index, num_nodes):
        return apply_directed_general_norm(
            dst_map,
            src_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
        )

    def message(self, z_src: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F, z_src)


class OrthogonalHetDirectedNSDConv(BaseHetDirectedNSDConv):
    """Orthogonal het directed NSD convolution."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        learn_alpha: bool = False,
        clamp_val: float = 10.0,
        use_attention: bool = False,
        orth_strategy: Literal["cayley", "fasth"] = "cayley",
    ):
        if orth_strategy not in {"cayley", "fasth"}:
            raise ValueError(
                f"orth_strategy must be 'cayley' or 'fasth', got {orth_strategy!r}"
            )
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            learn_alpha,
        )
        self.clamp_val = clamp_val
        self.use_attention = use_attention
        self.orth_strategy = orth_strategy

        if use_attention or orth_strategy == "fasth":
            num_params = stalk_dim * stalk_dim
        else:
            num_params = stalk_dim * (stalk_dim - 1) // 2

        self._make_map_generators(hidden_dim, num_params)
        self.reset_parameters()

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        params_dst, params_src = self._map_pair(
            x_feat, edge_index, node_types, edge_types
        )

        if self.use_attention:
            W_dst = attention_cayley(
                params_dst, self.stalk_dim, x_feat.device, x_feat.dtype
            )
            W_src = attention_cayley(
                params_src, self.stalk_dim, x_feat.device, x_feat.dtype
            )
        elif self.orth_strategy == "fasth":
            W_dst = householder(params_dst, self.stalk_dim)
            W_src = householder(params_src, self.stalk_dim)
        else:
            W_dst = cayley(params_dst, self.stalk_dim, self.clamp_val)
            W_src = cayley(params_src, self.stalk_dim, self.clamp_val)

        cross_map = W_dst.transpose(-2, -1) @ W_src
        return None, None, cross_map

    def _apply_norm(self, dst_map, src_map, cross_map, edge_index, num_nodes):
        return apply_directed_orthogonal_norm(cross_map, edge_index, num_nodes)

    def message(self, z_src: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F, z_src)


class LowRankHetDirectedNSDConv(BaseHetDirectedNSDConv):
    """Low-rank het directed NSD convolution."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        learn_alpha: bool = False,
        rank: int = 1,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            learn_alpha,
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self._make_map_generators(hidden_dim, 2 * stalk_dim * rank)
        self.reset_parameters()

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        raw_dst, raw_src = self._map_pair(x_feat, edge_index, node_types, edge_types)

        left_dst, right_dst = raw_dst.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )
        left_src, right_src = raw_src.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )

        gram_dst_dst = left_dst.transpose(-2, -1) @ left_dst  # [E, r, r]
        gram_src_src = left_src.transpose(-2, -1) @ left_src  # [E, r, r]
        gram_dst_src = left_dst.transpose(-2, -1) @ left_src  # [E, r, r]

        dst_map = right_dst @ gram_dst_dst @ right_dst.transpose(-2, -1)
        src_map = right_src @ gram_src_src @ right_src.transpose(-2, -1)
        cross_map = right_dst @ gram_dst_src @ right_src.transpose(-2, -1)
        return dst_map, src_map, cross_map

    def _apply_norm(self, dst_map, src_map, cross_map, edge_index, num_nodes):
        return apply_directed_low_rank_norm(
            dst_map,
            src_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
        )

    def message(self, z_src: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F, z_src)


__all__ = [
    "BaseHetDirectedNSDConv",
    "DiagonalHetDirectedNSDConv",
    "GeneralHetDirectedNSDConv",
    "OrthogonalHetDirectedNSDConv",
    "LowRankHetDirectedNSDConv",
]
