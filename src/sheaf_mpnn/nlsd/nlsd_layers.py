# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod
from typing import Literal

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.nlsd.nonlinearity import BaseEdgeNonlinearity
from sheaf_mpnn.utils import (
    ORTH_STRATEGIES,
    apply_diagonal_coboundary_norm,
    apply_general_coboundary_norm,
    apply_orthogonal_coboundary_norm,
    build_orthogonal,
    orth_param_count,
)


class BaseNLSDConv(BaseSheafConv):
    r"""Base class for Nonlinear Sheaf Diffusion convolutions (MPSNN form).

    Where :class:`~sheaf_mpnn.nsd.nsd_layers.BaseNSDConv` assembles the linear
    sheaf Laplacian from the products :math:`F_\mathrm{dst}^\top F_\mathrm{dst}`
    and :math:`F_\mathrm{dst}^\top F_\mathrm{src}`, NLSD cannot: the operator

    .. math::

        L^\Phi_\mathcal{F} = \delta^\top \circ \Phi \circ \delta

    has no matrix form once :math:`\Phi` is nonlinear.  The layer therefore
    keeps the two halves apart and evaluates them in sequence, one edge at a
    time (Eq. 13-14 of Zaghen et al., 2024):

    .. math::

        Y_t = \Phi\!\left(\delta_{\mathcal{F}(t)} D^{-1/2}
                          (I \otimes W_1^t) X_t W_2^t \right), \qquad
        X_{t+1} = (1 + \varepsilon) X_t
                  - \sigma\!\left(D^{-1/2} \delta^\top_{\mathcal{F}(t)} Y_t\right).

    Because :math:`L^\Phi_\mathcal{F}` is not a matrix, the usual symmetric
    normalisation :math:`D^{-1/2} L D^{-1/2}` cannot be applied to it.  Instead
    :math:`D^{-1/2}` is folded into the (linear) coboundary and its transpose,
    which recovers the standard normalised Laplacian when :math:`\Phi` is the
    identity.  Subclasses implement ``get_restriction_maps()``, returning the
    already-normalised endpoint maps, plus ``apply_map``/``apply_map_transpose``.

    Orientation.  Writing :math:`L^\Phi = \delta^\top \Phi \delta` requires
    picking an orientation for each undirected edge, and for a :math:`\Phi`
    that is not odd the result depends on that arbitrary choice.  This layer
    instead evaluates the node-wise definition (Eq. 8 of the paper) directly,

    .. math::

        L^\Phi_\mathcal{F}(x)_v = \sum_{u, v \lhd e} F_{v \lhd e}^\top\,
            \phi_e\!\left(F_{v \lhd e} x_v - F_{u \lhd e} x_u\right),

    in which each endpoint feeds :math:`\phi_e` the disagreement as *it* sees
    it.  ``edge_index`` already stores both directions of every edge, so this
    costs nothing extra and makes the layer permutation equivariant for any
    :math:`\Phi`.  Whenever :math:`\Phi` is odd -- the identity and both
    :math:`\psi'` gates -- the two orientations differ only by a sign that
    cancels, and this coincides exactly with :math:`\delta^\top \Phi \delta`.
    """

    map_generator: nn.Module

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
        sparse_learner: bool = False,
    ):
        r"""Initializes the shared NLSD convolution parameters.

        Args:
            stalk_dim: Stalk dimension ``d``. Node states have shape
                ``[stalk_dim, in_channels]``.
            in_channels: Feature dimension inside each stalk channel (f).
            hidden_dim: Unused by the reference-style linear map generator;
                kept for API compatibility with the sibling families.
            nonlinearity: The edge-space map :math:`\Phi`. Owned by the layer,
                so its parameters are learned per layer as in the reference.
            alpha: Initial diffusion step size.
            context_dim: Width of each node context vector ``x_feat``.
                Defaults to ``stalk_dim * in_channels``.
            add_self_loops: Whether to add self-loops before diffusing.
                Defaults to ``False``; the normalisation is already
                (deg+1)-augmented, so self-loops would augment twice. Self-loops
                carry a zero coboundary value, hence contribute nothing for any
                :math:`\Phi` with :math:`\Phi(0) = 0`.
            add_eps: If ``True``, adds the learnable per-stalk rescaling
                ``eps`` of Eq. 7/10, applied to the skip term as
                ``(1 + tanh(eps)) * x_stalk`` and initialised at zero.
            per_stalk_alpha: If ``True``, the diffusion term additionally
                carries the reference NLSD per-stalk gate
                ``(1 + tanh(alpha_gate))`` with ``alpha_gate`` initialised at
                -2 (so the update starts close to a pure skip connection).
                The threshold-gated variants use it; the learned-MLP variant
                does not.
            dropout: Dropout probability applied to the diffusion input; the
                skip path stays un-dropped, matching the reference.
            sparse_learner: Uses the reference stalk-summed map-generator input
                (LocalConcatSheafLearnerVariant), shrinking the learner input
                from ``2*d*f`` to ``2*f``.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            add_self_loops,
            add_lp=False,
            add_hp=False,
            sparse_learner=sparse_learner,
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

        The linear families normalise ``F^T F`` and ``F_dst^T F_src`` because
        those are the only quantities the assembled Laplacian needs. NLSD has
        to keep the two coboundary halves separate, so it normalises the
        endpoint maps in ``get_restriction_maps()`` instead.
        """
        raise NotImplementedError(
            "NLSD normalises endpoint maps via get_restriction_maps()."
        )

    @abstractmethod
    def get_restriction_maps(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Returns the D^{-1/2}-normalised endpoint restriction maps.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            edge_index: Graph connectivity ``[2, E]``.
            num_nodes: Number of nodes, for the degree scatter.

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
        self, x_feat: torch.Tensor, x_stalk: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Applies one nonlinear sheaf diffusion step to lifted node features.

        Args:
            x_feat: Node context features ``[num_nodes, context_dim]``.
            x_stalk: Lifted node features ``[num_nodes, d, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``.

        Returns:
            Updated stalk features ``[num_nodes, d, in_channels]``.
        """
        num_nodes = x_stalk.size(0)
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Dropout hits only the diffusion input; the skip term keeps the
        # un-dropped x_stalk, as in the reference update.
        z = self._apply_stalk_transform(self.dropout_layer(x_stalk))

        src_idx, dst_idx = edge_index
        map_dst, map_src = self.get_restriction_maps(x_feat, edge_index, num_nodes)

        # delta, oriented from the receiving node outwards: every directed edge
        # carries the disagreement as *its own* destination sees it. No global
        # orientation is chosen, so nothing depends on the node numbering.
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

        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        step = self.alpha * self.sigma(diffusion)
        if self.per_stalk_alpha:
            step = (1 + torch.tanh(self.alpha_gate)[None, :, None]) * step
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


class DiagonalNLSDConv(BaseNLSDConv):
    """NLSD convolution with diagonal restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
        sparse_learner: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            nonlinearity,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            per_stalk_alpha,
            dropout,
            sparse_learner,
        )
        self.map_generator = self._build_map_generator(stalk_dim)
        self.reset_parameters()

    def get_restriction_maps(self, x_feat, edge_index, num_nodes):
        """Diagonal maps ``w`` at both endpoints, scaled by ``(D+1)^{-1/2}``.

        Returns:
            ``(w_dst, w_src)``, both ``[E, d]`` -- diagonal entries rather than
            full matrices.
        """
        w_dst, w_src = self._bidirectional_input(x_feat, edge_index)
        return apply_diagonal_coboundary_norm(w_dst, w_src, edge_index, num_nodes)

    def apply_map(self, maps, stalk):
        # Diagonal maps act elementwise along the stalk axis.
        return maps[:, :, None] * stalk

    def apply_map_transpose(self, maps, stalk):
        # A diagonal matrix is its own transpose.
        return maps[:, :, None] * stalk


class OrthogonalNLSDConv(BaseNLSDConv):
    """NLSD convolution with orthogonal (bundle) restriction maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
        sparse_learner: bool = False,
        clamp_val: float = 10.0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        use_edge_weights: bool = False,
    ):
        """Initializes an orthogonal NLSD convolution layer.

        Args:
            stalk_dim: Stalk dimension and orthogonal map size. See
                :class:`BaseNLSDConv`.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Kept for API compatibility; unused by the generator.
            nonlinearity: The edge-space map Phi owned by this layer.
            alpha: Initial diffusion step size.
            context_dim: Width of each node context vector ``x_feat``.
            add_self_loops: Whether to add self-loops before diffusing.
            add_eps: Learnable per-stalk rescaling of the skip term.
            per_stalk_alpha: Per-stalk gate on the diffusion term.
            dropout: Dropout probability on the diffusion input.
            sparse_learner: Stalk-summed map-generator input.
            clamp_val: Maximum absolute value for the orthogonal-transform
                parameters. Defaults to 10.0.
            orth_strategy: Orthogonal parameterisation -- "cayley", "fasth",
                "fasthpp", or "householder". Defaults to "cayley".
            use_edge_weights: Learns the reference symmetric per-edge scalar
                that rescales the transport maps and replaces the combinatorial
                degree with the weighted one. Defaults to ``False``.

        See :class:`BaseNLSDConv` for the shared arguments.
        """
        if orth_strategy not in ORTH_STRATEGIES:
            raise ValueError(
                f"orth_strategy must be one of {ORTH_STRATEGIES}, got {orth_strategy!r}"
            )
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            nonlinearity,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            per_stalk_alpha,
            dropout,
            sparse_learner,
        )
        self.clamp_val = clamp_val
        self.orth_strategy = orth_strategy
        self.use_edge_weights = use_edge_weights
        if use_edge_weights:
            # Reference EdgeWeightLearner: bias-free scalar head, sigmoid at use.
            self.weight_generator = nn.Linear(2 * self.context_dim, 1, bias=False)
        self.map_generator = self._build_map_generator(
            orth_param_count(orth_strategy, stalk_dim)
        )
        self.reset_parameters()

    def get_restriction_maps(self, x_feat, edge_index, num_nodes):
        """Orthogonal maps at both endpoints, scaled by ``(deg+1)^{-1/2}``.

        Returns:
            ``(W_dst, W_src)``, both ``[E, d, d]``.
        """
        params_dst, params_src = self._bidirectional_input(x_feat, edge_index)
        w_dst = build_orthogonal(params_dst, self.d, self.orth_strategy, self.clamp_val)
        w_src = build_orthogonal(params_src, self.d, self.orth_strategy, self.clamp_val)

        edge_weight = None
        if self.use_edge_weights:
            # Symmetric per-edge scalar sigmoid(w_uv) * sigmoid(w_vu), so both
            # directed copies of an edge agree.
            w_a, w_b = self._bidirectional_pair(
                self.weight_generator, x_feat, edge_index
            )
            edge_weight = torch.sigmoid(w_a) * torch.sigmoid(w_b)

        return apply_orthogonal_coboundary_norm(
            w_dst, w_src, edge_index, num_nodes, edge_weight
        )

    def apply_map(self, maps, stalk):
        return torch.matmul(maps, stalk)

    def apply_map_transpose(self, maps, stalk):
        return torch.matmul(maps.transpose(-2, -1), stalk)


class _MatrixMapNLSDConv(BaseNLSDConv):
    r"""Shared machinery for the full-matrix restriction-map families.

    ``general`` and ``low_rank`` differ only in how the ``d x d`` map is
    parameterised -- unconstrained entries versus a rank-``r`` factorisation --
    so once the map is materialised the coboundary, its transpose and the block
    degree normalisation are identical.  Both *have* to materialise it: with
    :math:`\Phi` sitting between :math:`\delta` and :math:`\delta^\top` there is
    no assembled operator to keep in factored form, which is why this layer
    cannot borrow ``LowRankNSDConv``'s trick of composing the Gram matrices in
    :math:`r \times r` space.

    Neither family is part of the reference NLSD (paper Section 4.2 keeps only
    the diagonal and orthogonal ones); they are provided here because the
    framework's other families offer them and because the ``Phi = identity``
    parity tests extend to them unchanged.
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


class GeneralNLSDConv(_MatrixMapNLSDConv):
    r"""NLSD convolution with general (unconstrained ``d x d``) restriction maps.

    The most expressive family, at ``O(d^2)`` parameters per edge.  Note that
    only orthogonal maps preserve the norm of the disagreement
    :math:`F_v x_v - F_u x_u`, so the threshold nonlinearities -- which gate on
    that norm -- can be influenced by the shape of the maps themselves here.
    The learned-MLP :math:`\Phi` of the paper is indifferent to this.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
        sparse_learner: bool = False,
    ):
        """Initializes a general NLSD convolution layer.

        See :class:`BaseNLSDConv` for the shared arguments.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            nonlinearity,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            per_stalk_alpha,
            dropout,
            sparse_learner,
        )
        self.map_generator = self._build_map_generator(stalk_dim * stalk_dim)
        self.reset_parameters()

    def get_restriction_maps(self, x_feat, edge_index, num_nodes):
        """Full matrices at both endpoints, scaled by the block ``(D+I)^{-1/2}``.

        Returns:
            ``(F_dst, F_src)``, both ``[E, d, d]``.
        """
        f_dst, f_src = (
            w.view(-1, self.d, self.d)
            for w in self._bidirectional_input(x_feat, edge_index)
        )
        return self._normalise(f_dst, f_src, edge_index, num_nodes)


class LowRankNLSDConv(_MatrixMapNLSDConv):
    r"""NLSD convolution with low-rank restriction maps.

    Each map is parameterised as :math:`F = A B^\top` with
    :math:`A, B \in \mathbb{R}^{d \times r}`, costing ``2*d*r`` parameters per
    edge instead of ``d^2`` while keeping the full ``d``-dimensional stalk.

    The rank bound makes the sheaf degree :math:`D_v = \sum_e F^\top F` singular
    whenever ``rank * deg(v) < d``; the framework's augmented ``(D+I)^{-1/2}``
    is what keeps the normalisation defined there.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        nonlinearity: BaseEdgeNonlinearity,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        per_stalk_alpha: bool = True,
        dropout: float = 0.0,
        sparse_learner: bool = False,
        rank: int = 1,
    ):
        """Initializes a low-rank NLSD convolution layer.

        Args:
            stalk_dim: Stalk dimension ``d``. See :class:`BaseNLSDConv`.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Kept for API compatibility; unused by the generator.
            nonlinearity: The edge-space map Phi owned by this layer.
            alpha: Initial diffusion step size.
            context_dim: Width of each node context vector ``x_feat``.
            add_self_loops: Whether to add self-loops before diffusing.
            add_eps: Learnable per-stalk rescaling of the skip term.
            per_stalk_alpha: Per-stalk gate on the diffusion term.
            dropout: Dropout probability on the diffusion input.
            sparse_learner: Stalk-summed map-generator input.
            rank: Rank of each restriction map ``F = A B^T``. Must be positive.

        Raises:
            ValueError: If ``rank`` is not positive.
        """
        if rank <= 0:
            raise ValueError("rank must be positive")
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            nonlinearity,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            per_stalk_alpha,
            dropout,
            sparse_learner,
        )
        self.rank = rank
        self.map_generator = self._build_map_generator(2 * stalk_dim * rank)
        self.reset_parameters()

    def get_restriction_maps(self, x_feat, edge_index, num_nodes):
        """Rank-``r`` maps at both endpoints, scaled by the block ``(D+I)^{-1/2}``.

        Returns:
            ``(F_dst, F_src)``, both ``[E, d, d]`` -- the factors multiplied out,
            since Phi has to be applied between the two coboundary halves.
        """
        raw_dst, raw_src = self._bidirectional_input(x_feat, edge_index)

        # Reshape into [E, d, 2r], then split into left [E, d, r], right [E, d, r].
        left_dst, right_dst = raw_dst.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)
        left_src, right_src = raw_src.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)

        f_dst = torch.matmul(left_dst, right_dst.transpose(-2, -1))  # [E, d, d]
        f_src = torch.matmul(left_src, right_src.transpose(-2, -1))  # [E, d, d]
        return self._normalise(f_dst, f_src, edge_index, num_nodes)


__all__ = [
    "BaseNLSDConv",
    "DiagonalNLSDConv",
    "OrthogonalNLSDConv",
    "GeneralNLSDConv",
    "LowRankNLSDConv",
]
