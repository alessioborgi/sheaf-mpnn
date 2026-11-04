# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from hetsheaf_mpnn.base_conv import BaseHetSheafConv
from sheaf_mpnn.utils import (
    apply_diagonal_norm,
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
    attention_cayley,
    cayley,
    householder,
)


class BaseHetNSPConv(BaseHetSheafConv):
    """Base class for heterogeneous NSP convolutions.

    Implements the second-order NSP recurrence via state augmentation:
    node stalks are doubled to [x_curr; x_hist] in R^{2dxf}. Het context flags
    control what information feeds the restriction-map MLP, identical to the NSD
    counterpart. Returns the updated augmented state [N, 2d, f].

    The utilities ``_mlp_input_dim()``, ``_build_context_pair()``, and
    ``_ensemble_forward()`` are inherited from ``BaseHetSheafConv``.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            [in_channels],
            hidden_dim,
            context_dim,
            add_self_loops,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
        )
        self.add_eps = add_eps
        if add_eps:
            # Skip scale for x_curr: nominal coefficient 2 becomes (2 + eps_curr).
            self.eps_curr = nn.Parameter(torch.zeros(stalk_dim))
            # Scale factor for the history skip term: coeff becomes (1 + eps_hist).
            self.eps_hist = nn.Parameter(torch.zeros(stalk_dim))

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one het-NSP propagation step on the augmented state.

        Args:
            x_feat:     [N, context_dim]  Flattened current stalk (x_curr only).
            x_stalk:    [N, 2d, f]        Augmented stalks [x_curr; x_hist].
            edge_index: [2, E]            Graph connectivity.
            node_types: [N] or None
            edge_types: [E] or None

        Returns:
            Tensor [N, 2d, f]: Updated augmented stalks.
        """
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        x_curr = x_stalk[:, : self.stalk_dim, :]
        x_hist = x_stalk[:, self.stalk_dim :, :]

        z = self._apply_stalk_transform(x_curr)
        self_map, cross_map = self.get_map_products(
            x_feat, edge_index, node_types, edge_types
        )
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        src_idx, dst_idx = edge_index
        z_src, z_dst = z[src_idx], z[dst_idx]

        laplacian_out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )

        if self.add_eps:
            curr_scale = 2 + torch.tanh(self.eps_curr)[None, :, None]
            hist_scale = 1 + torch.tanh(self.eps_hist)[None, :, None]
            new_curr = (
                curr_scale * x_curr - hist_scale * x_hist - self.sigma(laplacian_out)
            )
        else:
            new_curr = 2 * x_curr - x_hist - self.sigma(laplacian_out)
        return torch.cat([new_curr, x_curr], dim=1)

    @abstractmethod
    def get_map_products(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""


class DiagonalHetNSPConv(BaseHetNSPConv):
    """Het NSP layer with diagonal restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            add_eps,
        )
        mlp_in = self._mlp_input_dim()
        if ensemble:
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(mlp_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, stalk_dim),
                    )
                    for _ in range(num_edge_types)
                ]
            )
        else:
            self.map_generator = nn.Sequential(
                nn.Linear(mlp_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, stalk_dim),
            )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        w_dst, w_src = raw.chunk(2, dim=0)
        return w_dst**2, w_dst * w_src

    def message(self, z_dst, z_src, self_map, cross_map):
        return self_map.unsqueeze(-1) * z_dst - cross_map.unsqueeze(-1) * z_src


class GeneralHetNSPConv(BaseHetNSPConv):
    """Het NSP layer with general (full dxd) restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            add_eps,
        )
        self.use_attention = use_attention
        mlp_in = self._mlp_input_dim()
        out_dim = stalk_dim * stalk_dim
        if ensemble:
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(mlp_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, out_dim),
                    )
                    for _ in range(num_edge_types)
                ]
            )
        else:
            self.map_generator = nn.Sequential(
                nn.Linear(mlp_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_general_norm(
            self_map, cross_map, edge_index, num_nodes, self.stalk_dim, self.training
        )

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        phi_dst, phi_src = (
            w.view(-1, self.stalk_dim, self.stalk_dim) for w in raw.chunk(2, dim=0)
        )
        if self.use_attention:
            eye = torch.eye(
                self.stalk_dim, device=x_feat.device, dtype=x_feat.dtype
            ).unsqueeze(0)
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)
        self_map = torch.matmul(phi_dst.transpose(-2, -1), phi_dst)
        cross_map = torch.matmul(phi_dst.transpose(-2, -1), phi_src)
        return self_map, cross_map


class OrthogonalHetNSPConv(BaseHetNSPConv):
    """Het NSP layer with orthogonal restriction maps (W^T W = I simplification)."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
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
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            add_eps,
        )
        self.clamp_val = clamp_val
        self.use_attention = use_attention
        self.orth_strategy = orth_strategy

        if use_attention or orth_strategy == "fasth":
            num_params = stalk_dim * stalk_dim
        else:
            num_params = (stalk_dim * (stalk_dim - 1)) // 2

        mlp_in = self._mlp_input_dim()
        if ensemble:
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(mlp_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, num_params),
                    )
                    for _ in range(num_edge_types)
                ]
            )
        else:
            self.map_generator = nn.Sequential(
                nn.Linear(mlp_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_params),
            )
        self.reset_parameters()

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        params_dst, params_src = raw.chunk(2, dim=0)

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

        cross_map = torch.matmul(W_dst.transpose(-2, -1), W_src)
        return None, cross_map

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def forward(self, x_feat, x_stalk, edge_index, node_types=None, edge_types=None):
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        x_curr = x_stalk[:, : self.stalk_dim, :]
        x_hist = x_stalk[:, self.stalk_dim :, :]

        z = self._apply_stalk_transform(x_curr)
        _, cross_map = self.get_map_products(x_feat, edge_index, node_types, edge_types)
        norm_self, norm_cross = self._apply_norm(None, cross_map, edge_index, num_nodes)
        src_idx, dst_idx = edge_index
        z_src, z_dst = z[src_idx], z[dst_idx]
        laplacian_out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )
        if self.add_eps:
            curr_scale = 2 + torch.tanh(self.eps_curr)[None, :, None]
            hist_scale = 1 + torch.tanh(self.eps_hist)[None, :, None]
            new_curr = (
                curr_scale * x_curr - hist_scale * x_hist - self.sigma(laplacian_out)
            )
        else:
            new_curr = 2 * x_curr - x_hist - self.sigma(laplacian_out)
        return torch.cat([new_curr, x_curr], dim=1)

    def message(self, z_dst, z_src, self_map, cross_map):
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class LowRankHetNSPConv(BaseHetNSPConv):
    """Het NSP layer with low-rank restriction maps F = left @ right^T."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        rank: int = 1,
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
            context_dim,
            add_self_loops,
            add_eps,
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        mlp_in = self._mlp_input_dim()
        out_dim = 2 * stalk_dim * rank
        if ensemble:
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(mlp_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, out_dim),
                    )
                    for _ in range(num_edge_types)
                ]
            )
        else:
            self.map_generator = nn.Sequential(
                nn.Linear(mlp_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_low_rank_norm(
            self_map, cross_map, edge_index, num_nodes, self.stalk_dim, self.training
        )

    def get_map_products(self, x_feat, edge_index, node_types, edge_types):
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        raw_dst, raw_src = raw.chunk(2, dim=0)

        left_dst, right_dst = raw_dst.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )
        left_src, right_src = raw_src.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )

        gram_dst_dst = torch.matmul(left_dst.transpose(-2, -1), left_dst)
        gram_dst_src = torch.matmul(left_dst.transpose(-2, -1), left_src)

        self_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_dst, right_dst.transpose(-2, -1))
        )
        cross_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_src, right_src.transpose(-2, -1))
        )
        return self_map, cross_map


__all__ = [
    "BaseHetNSPConv",
    "DiagonalHetNSPConv",
    "GeneralHetNSPConv",
    "OrthogonalHetNSPConv",
    "LowRankHetNSPConv",
]
