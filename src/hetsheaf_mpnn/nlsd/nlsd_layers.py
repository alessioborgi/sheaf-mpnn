# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod
from typing import Literal

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from hetsheaf_mpnn.base_conv import BaseHetSheafConv
from sheaf_mpnn.nlsd.nonlinearity import BaseEdgeNonlinearity
from sheaf_mpnn.utils import (
    ORTH_STRATEGIES,
    apply_diagonal_coboundary_norm,
    apply_general_coboundary_norm,
    apply_orthogonal_coboundary_norm,
    build_orthogonal,
    orth_param_count,
)


class BaseHetNLSDConv(BaseHetSheafConv):
    r"""Base class for heterogeneous Nonlinear Sheaf Diffusion convolutions.

    The het counterpart of
    :class:`~sheaf_mpnn.nlsd.nlsd_layers.BaseNLSDConv`: the diffusion operator
    is unchanged,

    .. math::

        L^\Phi_\mathcal{F}(x)_v = \sum_{u, v \lhd e} F_{v \lhd e}^\top\,
            \phi_e\!\left(F_{v \lhd e} x_v - F_{u \lhd e} x_u\right),

    but the restriction maps are generated from the *heterogeneous* context --
    node features, node types, edge types, or a per-edge-type ensemble of MLPs
    -- through the two-layer generator every het family shares, rather than the
    reference single bias-free linear of the homogeneous layer.

    Everything the nonlinearity sees is unchanged, so the :math:`\Phi` families
    in :mod:`sheaf_mpnn.nlsd.nonlinearity` are reused verbatim: they act on edge
    stalks and (symmetric functions of) the endpoint contexts, neither of which
    carries type information.

    As in the homogeneous layer, the node-wise form above is evaluated directly
    rather than :math:`\delta^\top \Phi \delta`, so no orientation has to be
    chosen and the layer stays permutation equivariant for any :math:`\Phi`.
    Subclasses implement ``get_restriction_maps()``, returning the
    already-normalised endpoint maps, plus ``apply_map``/``apply_map_transpose``.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
    ):
        r"""Initializes the shared het-NLSD convolution parameters.

        Args:
            stalk_dim: Stalk dimension ``d``.
            in_channels: Feature dimension inside each stalk channel (f).
            hidden_dim: Hidden width of the het map-generator MLP.
            nonlinearity: The edge-space map :math:`\Phi`, owned by this layer.
            alpha: Initial diffusion step size.
            num_node_types: Number of node types in the graph.
            num_edge_types: Number of edge types in the graph.
            use_features: Feed node features to the map generator.
            use_node_types: Feed one-hot node types to the map generator.
            use_edge_types: Feed one-hot edge types to the map generator.
            ensemble: Route each edge through its edge type's own generator MLP
                (TypeEnsemble strategy) instead of a single shared MLP.
            context_dim: Width of each node context vector ``x_feat``.
                Defaults to ``stalk_dim * in_channels``.
            add_self_loops: Whether to add self-loops before diffusing.
                Defaults to ``False``; the normalisation is already
                (deg+1)-augmented. Self-loops carry a zero coboundary value and
                so contribute nothing for any :math:`\Phi` with
                :math:`\Phi(0) = 0`.
            add_eps: Learnable per-stalk rescaling ``(1 + tanh(eps))`` of the
                skip term, initialised at zero.
            per_stalk_alpha: Whether the diffusion term carries the reference
                per-stalk ``(1 + tanh(alpha_gate))`` gate, ``alpha_gate``
                initialised at -2.
            dropout: Dropout probability applied to the diffusion input; the
                skip path stays un-dropped, matching the reference.
        """
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
        self.nonlinearity = nonlinearity
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.add_eps = add_eps
        if add_eps:
            self.eps = nn.Parameter(torch.zeros(self.stalk_dim))
        self.per_stalk_alpha = per_stalk_alpha
        if per_stalk_alpha:
            # Reference init: 1 + tanh(-2) ~= 0.036, a near-identity update.
            self.alpha_gate = nn.Parameter(-2.0 * torch.ones(self.stalk_dim))
        self.dropout_layer = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.add_eps:
            nn.init.zeros_(self.eps)
        if self.per_stalk_alpha:
            nn.init.constant_(self.alpha_gate, -2.0)
        self.nonlinearity.reset_parameters()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Unused: NLSD normalises the maps themselves, not their products.

        The linear het families normalise ``F^T F`` and ``F_dst^T F_src``
        because those are the only quantities the assembled Laplacian needs.
        NLSD has to keep the two coboundary halves separate, so it normalises
        the endpoint maps in ``get_restriction_maps()`` instead.
        """
        raise NotImplementedError(
            "Het NLSD normalises endpoint maps via get_restriction_maps()."
        )

    def _generate_maps(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the het map generator and splits the two endpoint perspectives.

        Returns the raw ``(dst, src)`` generator outputs, before any reshaping
        into matrices and before the degree normalisation.
        """
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        map_dst, map_src = raw.chunk(2, dim=0)
        return map_dst, map_src

    @abstractmethod
    def get_restriction_maps(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the D^{-1/2}-normalised endpoint restriction maps.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            edge_index: Graph connectivity ``[2, E]``.
            num_nodes: Number of nodes, for the degree scatter.
            node_types: Node-type indices ``[N]``, or ``None``.
            edge_types: Edge-type indices ``[E]``, or ``None``.

        Returns:
            ``(F_dst, F_src)`` -- the maps :math:`F_{\mathrm{dst} \lhd e}
            D_\mathrm{dst}^{-1/2}` and :math:`F_{\mathrm{src} \lhd e}
            D_\mathrm{src}^{-1/2}`, in whatever layout the variant uses.
        """

    @abstractmethod
    def apply_map(self, maps: torch.Tensor, stalk: torch.Tensor) -> torch.Tensor:
        """Applies per-edge restriction maps to per-edge stalks ``[E, d, f]``."""

    @abstractmethod
    def apply_map_transpose(
        self, maps: torch.Tensor, stalk: torch.Tensor
    ) -> torch.Tensor:
        """Applies the transposed restriction maps (the ``delta^T`` half)."""

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one heterogeneous nonlinear sheaf diffusion step.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            x_stalk: Lifted node features ``[N, d, in_channels]``.
            edge_index: Graph connectivity ``[2, E]``.
            node_types: Node-type indices ``[N]``, or ``None``.
            edge_types: Edge-type indices ``[E]``, or ``None``.

        Returns:
            Updated stalk features ``[N, d, in_channels]``.
        """
        num_nodes = x_stalk.size(0)
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            if edge_types is not None:
                edge_types = torch.cat([edge_types, edge_types.new_zeros(num_nodes)])

        # Dropout hits only the diffusion input; the skip term keeps the
        # un-dropped x_stalk, as in the reference update.
        z = self._apply_stalk_transform(self.dropout_layer(x_stalk))

        src_idx, dst_idx = edge_index
        map_dst, map_src = self.get_restriction_maps(
            x_feat, edge_index, num_nodes, node_types, edge_types
        )

        # delta, oriented from the receiving node outwards: every directed edge
        # carries the disagreement as *its own* destination sees it.
        edge_stalk = self.apply_map(map_dst, z[dst_idx]) - self.apply_map(
            map_src, z[src_idx]
        )
        # Phi: the nonlinearity lives entirely in edge space.
        edge_stalk = self.nonlinearity(edge_stalk, x_feat[dst_idx], x_feat[src_idx])

        # delta^T: pull back to the node stalks and sum over incident edges.
        diffusion = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            edge_stalk=edge_stalk,
            map_dst=map_dst,
            size=(num_nodes, num_nodes),
        )

        step = self.alpha * self.sigma(diffusion)
        if self.per_stalk_alpha:
            step = (1 + torch.tanh(self.alpha_gate)[None, :, None]) * step
        if not self.residual:
            # Return the bare diffusion message when the caller owns the residual.
            return -step
        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip - step

    def message(  # ty: ignore[invalid-method-override]
        self, edge_stalk: torch.Tensor, map_dst: torch.Tensor
    ) -> torch.Tensor:
        r"""Pulls each (already nonlinear) edge stalk back to its destination.

        Args:
            edge_stalk: Signed :math:`\Phi(\delta x)` per edge ``[E, d, f]``.
            map_dst: Normalised destination restriction maps.

        Returns:
            Per-edge messages ``[E, d, f]``.
        """
        return self.apply_map_transpose(map_dst, edge_stalk)


class DiagonalHetNLSDConv(BaseHetNLSDConv):
    """Het NLSD convolution with diagonal restriction maps."""

    def __init__(self, stalk_dim: int, in_channels: int, hidden_dim: int, **kwargs):
        super().__init__(stalk_dim, in_channels, hidden_dim, **kwargs)
        self._make_map_generators(hidden_dim, stalk_dim)
        self.reset_parameters()

    def get_restriction_maps(
        self, x_feat, edge_index, num_nodes, node_types, edge_types
    ):
        """Diagonal maps ``w`` at both endpoints, scaled by ``(D+1)^{-1/2}``.

        Returns:
            ``(w_dst, w_src)``, both ``[E, d]`` -- diagonal entries rather than
            full matrices.
        """
        w_dst, w_src = self._generate_maps(x_feat, edge_index, node_types, edge_types)
        return apply_diagonal_coboundary_norm(w_dst, w_src, edge_index, num_nodes)

    def apply_map(self, maps, stalk):
        # Diagonal maps act elementwise along the stalk axis.
        return maps[:, :, None] * stalk

    def apply_map_transpose(self, maps, stalk):
        # A diagonal matrix is its own transpose.
        return maps[:, :, None] * stalk


class OrthogonalHetNLSDConv(BaseHetNLSDConv):
    """Het NLSD convolution with orthogonal (bundle) restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        clamp_val: float = 10.0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        **kwargs,
    ):
        """Initializes an orthogonal het-NLSD convolution layer.

        Args:
            stalk_dim: Stalk dimension and orthogonal map size.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Hidden width of the het map-generator MLP.
            clamp_val: Maximum absolute value for the orthogonal-transform
                parameters. Defaults to 10.0.
            orth_strategy: Orthogonal parameterisation -- "cayley", "fasth",
                "fasthpp", or "householder". Defaults to "cayley".
            **kwargs: Shared arguments, see :class:`BaseHetNLSDConv`.

        Raises:
            ValueError: If ``orth_strategy`` is not a known parameterisation.
        """
        if orth_strategy not in ORTH_STRATEGIES:
            raise ValueError(
                f"orth_strategy must be one of {ORTH_STRATEGIES}, got {orth_strategy!r}"
            )
        super().__init__(stalk_dim, in_channels, hidden_dim, **kwargs)
        self.clamp_val = clamp_val
        self.orth_strategy = orth_strategy
        self._make_map_generators(
            hidden_dim, orth_param_count(orth_strategy, stalk_dim)
        )
        self.reset_parameters()

    def get_restriction_maps(
        self, x_feat, edge_index, num_nodes, node_types, edge_types
    ):
        """Orthogonal maps at both endpoints, scaled by ``(deg+1)^{-1/2}``.

        Returns:
            ``(W_dst, W_src)``, both ``[E, d, d]``.
        """
        params_dst, params_src = self._generate_maps(
            x_feat, edge_index, node_types, edge_types
        )
        w_dst = build_orthogonal(params_dst, self.d, self.orth_strategy, self.clamp_val)
        w_src = build_orthogonal(params_src, self.d, self.orth_strategy, self.clamp_val)
        return apply_orthogonal_coboundary_norm(w_dst, w_src, edge_index, num_nodes)

    def apply_map(self, maps, stalk):
        return torch.matmul(maps, stalk)

    def apply_map_transpose(self, maps, stalk):
        return torch.matmul(maps.transpose(-2, -1), stalk)


class _MatrixMapHetNLSDConv(BaseHetNLSDConv):
    r"""Shared machinery for the full-matrix het restriction-map families.

    ``general`` and ``low_rank`` differ only in how the ``d x d`` map is
    parameterised, so once it is materialised the coboundary, its transpose and
    the block degree normalisation are identical. Both *have* to materialise
    it: with :math:`\Phi` between :math:`\delta` and :math:`\delta^\top` there
    is no assembled operator to keep in factored form, so neither can borrow
    ``LowRankHetNSDConv``'s trick of composing the Gram matrices in
    :math:`r \times r` space.
    """

    def _normalise(
        self,
        map_dst: torch.Tensor,
        map_src: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Folds the block ``(D+I)^{-1/2}`` into both endpoint maps."""
        return apply_general_coboundary_norm(
            map_dst, map_src, edge_index, num_nodes, self.stalk_dim, self.training
        )

    def apply_map(self, maps, stalk):
        return torch.matmul(maps, stalk)

    def apply_map_transpose(self, maps, stalk):
        return torch.matmul(maps.transpose(-2, -1), stalk)


class GeneralHetNLSDConv(_MatrixMapHetNLSDConv):
    """Het NLSD convolution with general (unconstrained ``d x d``) maps."""

    def __init__(self, stalk_dim: int, in_channels: int, hidden_dim: int, **kwargs):
        super().__init__(stalk_dim, in_channels, hidden_dim, **kwargs)
        self._make_map_generators(hidden_dim, stalk_dim * stalk_dim)
        self.reset_parameters()

    def get_restriction_maps(
        self, x_feat, edge_index, num_nodes, node_types, edge_types
    ):
        """Full matrices at both endpoints, scaled by the block ``(D+I)^{-1/2}``.

        Returns:
            ``(F_dst, F_src)``, both ``[E, d, d]``.
        """
        f_dst, f_src = (
            w.view(-1, self.d, self.d)
            for w in self._generate_maps(x_feat, edge_index, node_types, edge_types)
        )
        return self._normalise(f_dst, f_src, edge_index, num_nodes)


class LowRankHetNLSDConv(_MatrixMapHetNLSDConv):
    r"""Het NLSD convolution with low-rank restriction maps.

    Each map is parameterised as :math:`F = A B^\top` with
    :math:`A, B \in \mathbb{R}^{d \times r}`, costing ``2*d*r`` generator
    outputs per edge instead of ``d^2`` while keeping the full ``d``-dimensional
    stalk. The rank bound makes the sheaf degree singular whenever
    ``rank * deg(v) < d``; the augmented ``(D+I)^{-1/2}`` keeps the
    normalisation defined there.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        rank: int = 1,
        **kwargs,
    ):
        """Initializes a low-rank het-NLSD convolution layer.

        Args:
            stalk_dim: Stalk dimension ``d``.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Hidden width of the het map-generator MLP.
            rank: Rank of each restriction map ``F = A B^T``. Must be positive.
            **kwargs: Shared arguments, see :class:`BaseHetNLSDConv`.

        Raises:
            ValueError: If ``rank`` is not positive.
        """
        if rank <= 0:
            raise ValueError("rank must be positive")
        super().__init__(stalk_dim, in_channels, hidden_dim, **kwargs)
        self.rank = rank
        self._make_map_generators(hidden_dim, 2 * stalk_dim * rank)
        self.reset_parameters()

    def get_restriction_maps(
        self, x_feat, edge_index, num_nodes, node_types, edge_types
    ):
        """Rank-``r`` maps at both endpoints, scaled by the block ``(D+I)^{-1/2}``.

        Returns:
            ``(F_dst, F_src)``, both ``[E, d, d]`` -- the factors multiplied
            out, since Phi has to be applied between the two coboundary halves.
        """
        raw_dst, raw_src = self._generate_maps(
            x_feat, edge_index, node_types, edge_types
        )

        # Reshape into [E, d, 2r], then split into left [E, d, r], right [E, d, r].
        left_dst, right_dst = raw_dst.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)
        left_src, right_src = raw_src.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)

        f_dst = torch.matmul(left_dst, right_dst.transpose(-2, -1))  # [E, d, d]
        f_src = torch.matmul(left_src, right_src.transpose(-2, -1))  # [E, d, d]
        return self._normalise(f_dst, f_src, edge_index, num_nodes)


__all__ = [
    "BaseHetNLSDConv",
    "DiagonalHetNLSDConv",
    "OrthogonalHetNLSDConv",
    "GeneralHetNLSDConv",
    "LowRankHetNLSDConv",
]
