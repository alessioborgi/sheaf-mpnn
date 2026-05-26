# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

from abc import abstractmethod

import torch
from torch import nn

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import (
    apply_diagonal_norm,
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
    attention_cayley,
    cayley,
    householder,
)


class BaseNSDConv(BaseSheafConv):
    """Base class for NSD convolutions (MPSNN formulation).

    The class computes the full sheaf Laplacian action inside message()
    without a separate scatter for the diagonal term.  Then, each of the
    subclasses implement get_map_products(), which precomputes the composed
    restriction-map products self_map = restriction_maps_dst^T restriction_maps_dst
    and cross_map = restriction_maps_dst^T restriction_maps_src per edge before the
    message loop, reducing message() to two matmuls instead of three.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = True,
    ):
        """Initializes the shared NSD convolution parameters.

        Args:
            stalk_dim: Stalk dimension. Each node state handled by the layer has shape
                ``[stalk_dim, in_channels]``.
            in_channels: Feature dimension inside each stalk channel (f).
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial residual diffusion step size.
            context_dim: Width of each node context vector ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
        """
        super().__init__(
            stalk_dim, in_channels, hidden_dim, context_dim, add_self_loops
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x_feat, x_stalk, edge_index):
        """Applies one NSD diffusion step to lifted node features.

        Args:
            x_feat (torch.Tensor): Node context features [num_nodes, context_dim].
            x_stalk (torch.Tensor): Lifted node features [num_nodes, d, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Updated stalk features [num_nodes, d, in_channels].
        """
        z = self._apply_stalk_transform(x_stalk)
        num_nodes = x_stalk.size(0)
        src_idx, dst_idx = edge_index

        self_map, cross_map = self.get_map_products(x_feat, edge_index)
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        z_src, z_dst = z[src_idx], z[dst_idx]
        laplacian_out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )

        return x_stalk - self.alpha * self.sigma(laplacian_out)

    @abstractmethod
    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""


class DiagonalNSDConv(BaseNSDConv):
    """Diagonal NSD convolution layer."""

    def __init__(
        self,
        stalk_dim,
        in_channels,
        hidden_dim,
        alpha=1.0,
        context_dim=None,
        add_self_loops: bool = True,
    ):
        super().__init__(
            stalk_dim, in_channels, hidden_dim, alpha, context_dim, add_self_loops
        )
        self.map_generator = nn.Sequential(
            nn.Linear(2 * self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_diagonal_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
        )

    def get_map_products(self, x_feat, edge_index):
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]

        # Batch both edge orientations into a single MLP forward pass then split.
        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        w_dst, w_src = self.map_generator(inp).chunk(2, dim=0)

        # Return element-wise products [E, d] instead of [E, d, d] diagonal matrices.
        return w_dst**2, w_dst * w_src

    def message(self, z_dst, z_src, self_map, cross_map):
        # self_map = w_dst^2 [E, d], cross_map = w_dst * w_src [E, d].
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src


class GeneralNSDConv(BaseNSDConv):
    """Generalized NSD convolution layer."""

    def __init__(
        self,
        stalk_dim,
        in_channels,
        hidden_dim,
        alpha=1.0,
        context_dim=None,
        add_self_loops: bool = True,
        use_attention: bool = False,
    ):
        super().__init__(
            stalk_dim, in_channels, hidden_dim, alpha, context_dim, add_self_loops
        )
        self.use_attention = use_attention
        self.map_generator = nn.Sequential(
            nn.Linear(2 * self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim * stalk_dim),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
        )

    def get_map_products(self, x_feat, edge_index):
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]

        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        phi_dst, phi_src = (
            w.view(-1, self.stalk_dim, self.stalk_dim)
            for w in self.map_generator(inp).chunk(2, dim=0)
        )

        if self.use_attention:
            eye = torch.eye(
                self.stalk_dim, device=x_feat.device, dtype=x_feat.dtype
            ).unsqueeze(0)
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)

        self_map = torch.matmul(phi_dst.transpose(-2, -1), phi_dst)  # [E, d, d]
        cross_map = torch.matmul(phi_dst.transpose(-2, -1), phi_src)  # [E, d, d]

        return self_map, cross_map


class OrthogonalNSDConv(BaseNSDConv):
    """Orthogonal NSD convolution layer."""

    def __init__(
        self,
        stalk_dim,
        in_channels,
        hidden_dim,
        alpha=1.0,
        context_dim=None,
        add_self_loops: bool = True,
        clamp_val=10.0,
        use_attention: bool = False,
        orth_strategy="cayley",
    ):
        """Initializes an orthogonal NSD convolution layer.

        The map generator outputs parameters for either: (1) entries of a
        skew-symmetric matrix (cayley), (2) Householder vectors (fasth),
        or (3) attention-based mappings. All parameterisations produce
        orthogonal ``d x d`` restriction maps.

        Args:
            stalk_dim (int): Stalk dimension and orthogonal restriction-map matrix size.
            in_channels (int): Feature dimension inside each stalk channel.
            hidden_dim (int): Hidden width of the restriction-map generator MLP.
            alpha (float, optional): Initial learnable diffusion step size. Defaults
                to 1.0.
            context_dim (int, optional): Width of ``x_feat``. Defaults to
                ``d * in_channels`` when omitted.
            add_self_loops (bool, optional): If ``True``, self-loops augment the degree
                used for normalization. Defaults to ``True``.
            clamp_val (float, optional): Maximum absolute value for clamping
                Cayley-transform parameters. Defaults to 10.0.
            use_attention (bool, optional): If ``True``, uses the attention-based
                Cayley initialization from main. Defaults to ``False``.
            orth_strategy (str, optional): "cayley" or "fasth". Defaults to "cayley".
        """
        super().__init__(
            stalk_dim, in_channels, hidden_dim, alpha, context_dim, add_self_loops
        )
        self.clamp_val = clamp_val
        self.use_attention = use_attention
        self.orth_strategy = orth_strategy

        if use_attention or orth_strategy == "fasth":
            num_params = stalk_dim * stalk_dim
        else:
            num_params = (stalk_dim * (stalk_dim - 1)) // 2

        self.map_generator = nn.Sequential(
            nn.Linear(2 * self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index):
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]

        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        params_dst, params_src = self.map_generator(inp).chunk(2, dim=0)

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

        # W_dst^T W_dst = I (orthogonality), so self_map = I; only cross_map is needed.
        cross_map = torch.matmul(W_dst.transpose(-2, -1), W_src)  # [E, d, d]
        return None, cross_map

    def forward(self, x_feat, x_stalk, edge_index):
        z = self._apply_stalk_transform(x_stalk)
        num_nodes = x_stalk.size(0)
        src_idx, dst_idx = edge_index
        _, cross_map = self.get_map_products(x_feat, edge_index)

        norm_self, norm_cross = self._apply_norm(None, cross_map, edge_index, num_nodes)

        z_src, z_dst = z[src_idx], z[dst_idx]
        laplacian_out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )
        return x_stalk - self.alpha * self.sigma(laplacian_out)

    def message(self, z_dst, z_src, self_map, cross_map):
        # self_map = D^{-1}_dst [E,1,1]; cross_map = D^{-1/2}_dst W^T W D^{-1/2}_src.
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class LowRankNSDConv(BaseNSDConv):
    """Low-rank NSD convolution layer.

    Parameterizes each restriction map as F = A @ B^T where A, B ∈ R^{dxr},
    bounding the effective rank of each map to at most ``rank``. This gives
    cheaper parameterization than ``GeneralNSDConv`` (2*d*r vs d*d params per
    map) while keeping the full d-dimensional stalk.

    The Laplacian products reduce to:
        restriction_maps_dst^T restriction_maps_dst
            = right_dst (left_dst^T left_dst) right_dst^T  [d, d]
        restriction_maps_dst^T restriction_maps_src
            = right_dst (left_dst^T left_src) right_src^T  [d, d]
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        rank: int = 1,
    ):
        """Initializes a low-rank NSD convolution layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
            rank: Rank of each restriction map F = A @ B^T. Must be positive.
        """
        super().__init__(
            stalk_dim, in_channels, hidden_dim, alpha, context_dim, add_self_loops
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.map_generator = nn.Sequential(
            nn.Linear(2 * self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stalk_dim * rank),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_low_rank_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
        )

    def get_map_products(self, x_feat, edge_index):
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]

        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        raw_dst, raw_src = self.map_generator(inp).chunk(2, dim=0)

        # Reshape into [E, d, 2r], then split into left [E, d, r] and right [E, d, r].
        # Restriction map F = left @ right^T, so F^T = right @ left^T.
        left_dst, right_dst = raw_dst.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )
        left_src, right_src = raw_src.view(-1, self.stalk_dim, 2 * self.rank).chunk(
            2, dim=-1
        )

        gram_dst_dst = torch.matmul(left_dst.transpose(-2, -1), left_dst)  # [E, r, r]
        gram_dst_src = torch.matmul(left_dst.transpose(-2, -1), left_src)  # [E, r, r]

        self_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_dst, right_dst.transpose(-2, -1))
        )
        cross_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_src, right_src.transpose(-2, -1))
        )
        return self_map, cross_map


__all__ = [
    "BaseNSDConv",
    "DiagonalNSDConv",
    "GeneralNSDConv",
    "OrthogonalNSDConv",
    "LowRankNSDConv",
]
