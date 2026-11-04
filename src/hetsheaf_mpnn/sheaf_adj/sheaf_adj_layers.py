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


class BaseHetSheafAdjConv(BaseHetSheafConv):
    r"""Base class for heterogeneous SheafAdjacency convolutions.

    Extends :class:`~hetsheaf_mpnn.base_conv.BaseHetSheafConv` with the adjacency
    propagation rule from
    :class:`~sheaf_mpnn.sheaf_adj.sheaf_adj_layers.BaseSheafAdjConv`:

    .. math::

        \mathbf{x}_u^{(k+1)} = (1 + \boldsymbol{\varepsilon}) \odot \mathbf{x}_u^{(k)}
        + \alpha\,\sigma\!\left(\sum_{v \sim u} \hat{A}_{\mathcal{F},uv}\,
        \mathbf{z}_v^{(k)}\right).

    Heterogeneous context is injected into the restriction-map MLP via the flags
    ``use_features``, ``use_node_types``, ``use_edge_types``, and ``ensemble``.
    The utilities ``_mlp_input_dim()``, ``_build_context_pair()``, and
    ``_ensemble_forward()`` are inherited from
    :class:`~hetsheaf_mpnn.base_conv.BaseHetSheafConv`.

    Concrete subclasses must implement ``get_map_products()``.
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
        add_eps: bool = False,
    ):
        """Initializes shared het-SheafAdjacency convolution parameters.

        Args:
            stalk_dim: Stalk dimension d.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            num_node_types: Number of node types (used when ``use_node_types=True``
                or ``ensemble=True``).
            num_edge_types: Number of edge types (used when ``use_edge_types=True``
                or ``ensemble=True``).
            use_features: If ``True``, node features are part of the MLP context.
            use_node_types: If ``True``, node-type embeddings are appended to the
                MLP context.
            use_edge_types: If ``True``, edge-type embeddings are appended to the
                MLP context.
            ensemble: If ``True``, uses one MLP per edge type instead of a single
                shared MLP.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
        """
        # Wrap scalar in_channels into a single-element list for BaseHetSheafConv.
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
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.add_eps = add_eps
        if add_eps:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one het-SheafAdjacency propagation step.

        Args:
            x_feat:     [N, context_dim]             Flattened stalk (context for maps).
            x_stalk:    [N, stalk_dim, in_channels]  Structured stalk to update.
            edge_index: [2, E]                       Graph connectivity.
            node_types: [N] LongTensor or None       Node-type indices.
            edge_types: [E] LongTensor or None       Edge-type indices.

        Returns:
            Updated stalk [N, stalk_dim, in_channels].
        """
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                # Pad edge_types with zeros (type-0 self-loops) to match new edges.
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        z = self._apply_stalk_transform(x_stalk)
        self_map, cross_map = self.get_map_products(
            x_feat, edge_index, node_types, edge_types
        )
        # Only norm_cross enters the message; norm_self is used by _apply_norm
        # to compute D^{-1/2} but not propagated further.
        _, norm_cross = self._apply_norm(self_map, cross_map, edge_index, num_nodes)

        src_idx, _dst_idx = edge_index
        z_src = z[src_idx]

        # Aggregate A_hat_F z: sum_{v~u} D_u^{-1/2} F_ue^T F_ve D_v^{-1/2} z_v.
        adj_agg = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_src=z_src,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )
        # x^{k+1} = (1+eps)*x + alpha * sigma([A_hat z]_u).
        msg = self.alpha * self.sigma(adj_agg)
        if not self.residual:
            # Return the bare adjacency message when the caller owns the residual.
            return msg
        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip + msg

    def message(  # ty: ignore[invalid-method-override]
        self, z_src: torch.Tensor, cross_map: torch.Tensor
    ) -> torch.Tensor:
        r"""Builds per-edge adjacency messages.

        Args:
            z_src: Source-node transformed stalks [E, d, f].
            cross_map: Normalised :math:`D_u^{-1/2} F_\mathrm{dst}^\top F_\mathrm{src}
                D_v^{-1/2}` per edge [E, d, d].

        Returns:
            torch.Tensor: Per-edge adjacency messages [E, d, f].
        """
        return torch.matmul(cross_map, z_src)

    @abstractmethod
    def get_map_products(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""


class DiagonalHetSheafAdjConv(BaseHetSheafAdjConv):
    r"""Het SheafAdjacency layer with diagonal restriction maps.

    Each restriction map is a :math:`d \times d` diagonal matrix represented as a
    length-:math:`d` vector :math:`\mathbf{w}`.  The map products are:

    .. math::

        \texttt{self\_map} = \mathbf{w}_\mathrm{dst}^2, \quad
        \texttt{cross\_map} = \mathbf{w}_\mathrm{dst} \odot \mathbf{w}_\mathrm{src}
        \in \mathbb{R}^d.
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
        add_eps: bool = False,
    ):
        """Initializes a diagonal het-SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension d; restriction maps are length-d vectors.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            num_node_types: Number of distinct node types.
            num_edge_types: Number of distinct edge types.
            use_features: Include node features in the MLP context.
            use_node_types: Append node-type embeddings to the MLP context.
            use_edge_types: Append edge-type embeddings to the MLP context.
            ensemble: One MLP per edge type when ``True``.
            context_dim: Width of ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
        """
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
        """Compute diagonal map products for all edges.

        Args:
            x_feat: Flattened stalk context [N, context_dim].
            edge_index: Graph connectivity [2, E].
            node_types: Node-type indices [N] or None.
            edge_types: Edge-type indices [E] or None.

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d] -- element-wise products
            ``w_dst**2`` and ``w_dst * w_src`` respectively.
        """
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        w_dst, w_src = raw.chunk(2, dim=0)
        # Return element-wise products [E, d] instead of [E, d, d] diagonal matrices.
        return w_dst**2, w_dst * w_src

    def message(self, z_src, cross_map):
        """Builds diagonal adjacency messages.

        Args:
            z_src: Source-node transformed stalks [E, d, f].
            cross_map: Normalised ``w_dst * w_src`` per edge [E, d].

        Returns:
            torch.Tensor: Per-edge messages [E, d, f].
        """
        # cross_map = (normalised w_dst * w_src) [E, d]; broadcast over feature dim.
        return cross_map.unsqueeze(-1) * z_src


class GeneralHetSheafAdjConv(BaseHetSheafAdjConv):
    r"""Het SheafAdjacency layer with general
    (full :math:`d \times d`) restriction maps.

    Each restriction map :math:`F \in \mathbb{R}^{d \times d}` is unconstrained.
    The map products are :math:`F_\mathrm{dst}^\top F_\mathrm{dst}` (self) and
    :math:`F_\mathrm{dst}^\top F_\mathrm{src}` (cross).

    When ``use_attention=True``, the raw MLP output is processed as
    :math:`I - \text{softmax}(\Phi)`, giving near-zero Gram products at
    initialisation.
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
        use_attention: bool = False,
        add_eps: bool = False,
    ):
        """Initializes a general (full d x d) het-SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension d; restriction maps are d x d matrices.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            num_node_types: Number of distinct node types.
            num_edge_types: Number of distinct edge types.
            use_features: Include node features in the MLP context.
            use_node_types: Append node-type embeddings to the MLP context.
            use_edge_types: Append edge-type embeddings to the MLP context.
            ensemble: One MLP per edge type when ``True``.
            context_dim: Width of ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
            use_attention: Apply ``I - softmax(Phi)`` post-processing for near-zero
                Gram products at initialisation.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
        """
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
        """Compute general map products F_dst^T F_dst and F_dst^T F_src for all edges.

        Args:
            x_feat: Flattened stalk context [N, context_dim].
            edge_index: Graph connectivity [2, E].
            node_types: Node-type indices [N] or None.
            edge_types: Edge-type indices [E] or None.

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d, d].
        """
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
            # softmax gives row-stochastic phi; I - phi has zero row sums,
            # so its Gram products have sheaf-Laplacian structure.
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)
        self_map = torch.matmul(phi_dst.transpose(-2, -1), phi_dst)  # [E, d, d]
        cross_map = torch.matmul(phi_dst.transpose(-2, -1), phi_src)  # [E, d, d]
        return self_map, cross_map


class OrthogonalHetSheafAdjConv(BaseHetSheafAdjConv):
    r"""Het SheafAdjacency layer with orthogonal restriction maps.

    Follows the same pattern as the homogeneous
    :class:`~sheaf_mpnn.sheaf_adj.sheaf_adj_layers.OrthogonalSheafAdjConv`:
    het context feeds the orthogonal-map generator (cayley / fasth / attention);
    :math:`W^\top W = I` means the self-map is the identity, so only the
    cross-map :math:`W_\mathrm{dst}^\top W_\mathrm{src}` is used.

    ``forward()`` is overridden to pass ``self_map=None`` to
    :func:`~sheaf_mpnn.utils.apply_orthogonal_norm`, which uses scalar degree values
    rather than per-edge matrix inverses.
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
        add_eps: bool = False,
    ):
        """Initializes an orthogonal het-SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension d; restriction maps are d x d
                orthogonal matrices.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            num_node_types: Number of distinct node types.
            num_edge_types: Number of distinct edge types.
            use_features: Include node features in the MLP context.
            use_node_types: Append node-type embeddings to the MLP context.
            use_edge_types: Append edge-type embeddings to the MLP context.
            ensemble: One MLP per edge type when ``True``.
            context_dim: Width of ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
            clamp_val: Clamp threshold for Cayley parameters. Defaults to 10.0.
            use_attention: Use attention-weighted Cayley map. Defaults to False.
            orth_strategy: Orthogonalisation strategy: ``"cayley"`` or ``"fasth"``
                (Householder). Defaults to ``"cayley"``.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
        """
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
        """Compute orthogonal cross-map W_dst^T W_src for all edges.

        Args:
            x_feat: Flattened stalk context [N, context_dim].
            edge_index: Graph connectivity [2, E].
            node_types: Node-type indices [N] or None.
            edge_types: Edge-type indices [E] or None.

        Returns:
            tuple: ``(None, cross_map)`` where ``cross_map`` is [E, d, d].
            ``self_map`` is ``None`` because W^T W = I.
        """
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

        # W_dst^T W_dst = I; return None as self_map placeholder.
        cross_map = torch.matmul(W_dst.transpose(-2, -1), W_src)  # [E, d, d]
        return None, cross_map

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Apply degree normalisation for orthogonal maps (self_map is None)."""
        # self_map is None; apply_orthogonal_norm uses scalar degree values only.
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def forward(self, x_feat, x_stalk, edge_index, node_types=None, edge_types=None):
        """Applies one orthogonal het-SheafAdjacency step (self_map = None)."""
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        z = self._apply_stalk_transform(x_stalk)
        _, cross_map = self.get_map_products(x_feat, edge_index, node_types, edge_types)
        _, norm_cross = self._apply_norm(None, cross_map, edge_index, num_nodes)
        src_idx, _dst_idx = edge_index
        z_src = z[src_idx]
        adj_agg = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_src=z_src,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )
        msg = self.alpha * self.sigma(adj_agg)
        if not self.residual:
            # Return the bare adjacency message when the caller owns the residual.
            return msg
        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip + msg


class LowRankHetSheafAdjConv(BaseHetSheafAdjConv):
    r"""Het SheafAdjacency layer with low-rank restriction maps F = A @ B^T (rank r).

    Cheaper parameterization than :class:`GeneralHetSheafAdjConv`
    (:math:`2dr` vs :math:`d^2` parameters per map) while keeping the full
    :math:`d`-dimensional stalk.  See
    :class:`~sheaf_mpnn.sheaf_adj.sheaf_adj_layers.LowRankSheafAdjConv` for the
    map-product formulae.
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
        rank: int = 1,
        add_eps: bool = False,
    ):
        """Initializes a low-rank het-SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            num_node_types: Number of distinct node types.
            num_edge_types: Number of distinct edge types.
            use_features: Include node features in the MLP context.
            use_node_types: Append node-type embeddings to the MLP context.
            use_edge_types: Append edge-type embeddings to the MLP context.
            ensemble: One MLP per edge type when ``True``.
            context_dim: Width of ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
            rank: Rank r of each restriction map F = A @ B^T. Must be positive.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
        """
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
        """Compute low-rank map products F_dst^T F_dst and F_dst^T F_src for all edges.

        Each restriction map F = A @ B^T (shape [d, d], rank r) so
        F^T F = B (A^T A) B^T.  Only the rank-r Gram matrices are materialised
        before the final [E, d, d] outer product.

        Args:
            x_feat: Flattened stalk context [N, context_dim].
            edge_index: Graph connectivity [2, E].
            node_types: Node-type indices [N] or None.
            edge_types: Edge-type indices [E] or None.

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d, d].
        """
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        raw_dst, raw_src = raw.chunk(2, dim=0)

        # Reshape into [E, d, 2r], split into left [E, d, r] and right [E, d, r].
        # F = left @ right^T, so F^T F = right (left^T left) right^T.
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
    "BaseHetSheafAdjConv",
    "DiagonalHetSheafAdjConv",
    "GeneralHetSheafAdjConv",
    "OrthogonalHetSheafAdjConv",
    "LowRankHetSheafAdjConv",
]
