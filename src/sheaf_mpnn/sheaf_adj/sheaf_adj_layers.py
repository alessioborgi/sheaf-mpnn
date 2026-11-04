# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod
from typing import Literal

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import (
    ORTH_STRATEGIES,
    apply_diagonal_norm,
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
    attention_cayley,
    build_orthogonal,
    orth_param_count,
)


class BaseSheafAdjConv(BaseSheafConv):
    r"""Base class for SheafAdjacency convolutions (SheafMPNN formulation).

    Propagates stalk features through the normalised sheaf adjacency matrix
    :math:`\hat{A}_\mathcal{F}` instead of the sheaf Laplacian
    :math:`\Delta_\mathcal{F}` (Bourgerie et al., "Deep Neural Sheaf Diffusion",
    Eq. 10-11, adjacency-operator formulation).  The update rule is

    .. math::

        \mathbf{x}_u^{(k+1)} = (1 + \boldsymbol{\varepsilon}) \odot \mathbf{x}_u^{(k)}
        + \alpha\,\sigma\!\left(\sum_{v \sim u} \hat{A}_{\mathcal{F},uv}\,
        \mathbf{z}_v^{(k)}\right),

    where :math:`\mathbf{z}_v^{(k)} = W_1^{(k)}\mathbf{x}_v^{(k)}W_2^{(k)}` is the
    bilateral stalk transform and
    :math:`\hat{A}_{\mathcal{F},uv} = D_u^{-1/2} F_{u\trianglelefteq e}^\top
    F_{v\trianglelefteq e} D_v^{-1/2}` is the :math:`(u,v)`-block of the normalised
    sheaf adjacency matrix.

    This class differs from :class:`~sheaf_mpnn.nsd.nsd_layers.BaseNSDConv` in two
    ways:

    * ``message()`` returns only the **cross-map** term
      :math:`\hat{A}_{\mathcal{F},uv}\mathbf{z}_v` (adjacency transport of the
      neighbour), without the self-map subtraction that forms the Laplacian.
    * The self term is **dropped**, not relocated: the nonlinearity sees the
      raw aggregate, :math:`\alpha\,\sigma(\text{agg})`, with no
      :math:`-\mathbf{z}_u` inside :math:`\sigma(\cdot)`. Subtracting
      :math:`\mathbf{z}_u` back in there would make this operator algebraically
      identical to :math:`-\Delta_\mathcal{F}\mathbf{z}` (since
      :math:`\hat{A}_\mathcal{F} = I - \Delta_\mathcal{F}`), defeating the purpose
      of the adjacency formulation: keeping the signal fed to :math:`\sigma`
      from vanishing as neighbouring stalks reach agreement.

    Each concrete subclass implements ``get_map_products()``, which precomputes the
    composed restriction-map products

    .. math::

        \texttt{self\_map} = F_\mathrm{dst}^\top F_\mathrm{dst}, \qquad
        \texttt{cross\_map} = F_\mathrm{dst}^\top F_\mathrm{src}

    per edge before the message loop.  ``_apply_norm()`` then applies the symmetric
    :math:`D^{-1/2}` normalisation to both products.
    """

    map_generator: nn.Module

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initializes shared SheafAdjacency convolution parameters.

        Args:
            stalk_dim: Stalk dimension d. Each node state handled by the layer has
                shape ``[stalk_dim, in_channels]``.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of each node context vector ``x_feat``. Defaults to
                ``stalk_dim * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
                Defaults to ``True``.
            add_eps: If ``True``, adds a learnable per-stalk rescaling
                ``eps in R^d`` (Remark 4, Appendix A.1).  The skip term
                ``x_stalk`` is replaced by ``(1 + eps) * x_stalk``, where eps
                is broadcast across the feature channel dimension.  Initialized
                to zero so default behaviour is the standard SheafAdj update.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            add_self_loops,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.add_eps = add_eps
        if add_eps:
            self.eps = nn.Parameter(torch.zeros(self.stalk_dim))

    def forward(
        self, x_feat: torch.Tensor, x_stalk: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Applies one SheafAdjacency propagation step to lifted node features.

        Args:
            x_feat (torch.Tensor): Node context features [num_nodes, context_dim].
            x_stalk (torch.Tensor): Lifted node features [num_nodes, d, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Updated stalk features [num_nodes, d, in_channels].
        """
        z = self._apply_stalk_transform(x_stalk)
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        src_idx, dst_idx = edge_index  # noqa: F841

        self_map, cross_map = self.get_map_products(x_feat, edge_index)
        # Only norm_cross is used in message(); norm_self is consumed by _apply_norm
        # to build D^{-1/2} but not forwarded into propagate.
        _, norm_cross = self._apply_norm(self_map, cross_map, edge_index, num_nodes)

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

        Unlike the Laplacian message (which subtracts a self term), this passes
        only the transported neighbour signal:
        :math:`\hat{A}_{\mathcal{F},uv}\,\mathbf{z}_v =
        D_u^{-1/2} F_{u\trianglelefteq e}^\top F_{v\trianglelefteq e}
        D_v^{-1/2}\,\mathbf{z}_v`.

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
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""


class DiagonalSheafAdjConv(BaseSheafAdjConv):
    r"""SheafAdjacency layer with diagonal restriction maps.

    Each restriction map is a :math:`d \times d` diagonal matrix represented as a
    length-:math:`d` vector :math:`\mathbf{w}`.  The precomputed map products
    are:

    .. math::

        \texttt{self\_map} = \mathbf{w}_\mathrm{dst}^2, \quad
        \texttt{cross\_map} = \mathbf{w}_\mathrm{dst} \odot \mathbf{w}_\mathrm{src}
        \in \mathbb{R}^d.

    The map generator MLP outputs :math:`d` scalars per edge orientation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.map_generator = nn.Sequential(
            nn.Linear(self._generator_in_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Apply diagonal degree normalisation to self- and cross-map products."""
        return apply_diagonal_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
        )

    def get_map_products(self, x_feat, edge_index):
        """Compute diagonal map products w_dst**2 and w_dst*w_src for all edges.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d] -- element-wise products.
        """
        # Batch both edge orientations into a single MLP forward pass then split.
        w_dst, w_src = self._bidirectional_input(x_feat, edge_index)

        # Element-wise products [E, d], extended with fixed lp/hp channels.
        return self._append_fixed_diag(w_dst**2, w_dst * w_src)

    def message(self, z_src: torch.Tensor, cross_map: torch.Tensor) -> torch.Tensor:
        """Builds diagonal adjacency messages.

        Args:
            z_src: Source-node transformed stalks [E, d, f].
            cross_map: Normalised w_dst*w_src per edge [E, d].

        Returns:
            torch.Tensor: Per-edge adjacency messages [E, d, f].
        """
        # cross_map = (normalised w_dst * w_src) [E, d]; broadcast over feature dim.
        return cross_map[:, :, None] * z_src


class GeneralSheafAdjConv(BaseSheafAdjConv):
    r"""SheafAdjacency layer with general (full :math:`d \times d`) restriction maps.

    Each restriction map :math:`F \in \mathbb{R}^{d \times d}` is an unconstrained
    square matrix.  The map products are:

    .. math::

        \texttt{self\_map} = F_\mathrm{dst}^\top F_\mathrm{dst},\quad
        \texttt{cross\_map} = F_\mathrm{dst}^\top F_\mathrm{src}
        \in \mathbb{R}^{d \times d}.

    When ``use_attention=True`` the raw MLP output is passed through a softmax
    to produce a row-stochastic matrix, and the restriction map is set to
    :math:`I - \text{softmax}(\Phi)`, so the Gram products have sheaf-Laplacian
    structure near initialisation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        r"""Initializes a general SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension f.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of ``x_feat``.
            add_self_loops: If ``True``, self-loops augment the degree normalization.
            use_attention: If ``True``, uses the softmax-based near-identity map
                initialisation :math:`F = I - \\text{softmax}(\\Phi)`.
                Defaults to ``False``.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.use_attention = use_attention
        self.map_generator = nn.Sequential(
            nn.Linear(self._generator_in_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim * stalk_dim),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Apply general degree normalisation to self- and cross-map products."""
        return apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
            learned_dim=self.d,
        )

    def get_map_products(self, x_feat, edge_index):
        """Compute general map products F_dst^T F_dst and F_dst^T F_src for all edges.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d, d].
        """
        phi_dst, phi_src = (
            w.view(-1, self.d, self.d)
            for w in self._bidirectional_input(x_feat, edge_index)
        )

        if self.use_attention:
            eye = torch.eye(self.d, device=x_feat.device, dtype=x_feat.dtype).unsqueeze(
                0
            )
            # softmax gives row-stochastic phi; I - phi has zero row sums,
            # so its Gram products have sheaf-Laplacian structure.
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)

        self_map = torch.matmul(phi_dst.transpose(-2, -1), phi_dst)  # [E, d, d]
        cross_map = torch.matmul(phi_dst.transpose(-2, -1), phi_src)  # [E, d, d]

        return self._append_fixed_block(self_map, cross_map)


class OrthogonalSheafAdjConv(BaseSheafAdjConv):
    r"""SheafAdjacency layer with orthogonal restriction maps.

    Each restriction map :math:`W` satisfies :math:`W^\top W = I_d`.  Because
    :math:`F^\top F = I`, the sheaf degree simplifies to
    :math:`D_u = \deg(u) \cdot I_d`, so :math:`D_u^{-1/2} = \deg(u)^{-1/2} I_d`.
    The self-map product :math:`F_\mathrm{dst}^\top F_\mathrm{dst} = I` is returned
    as a placeholder; the actual normalisation is handled by
    :func:`~sheaf_mpnn.utils.apply_orthogonal_norm` using scalar degree values only.

    The cross-map is :math:`W_\mathrm{dst}^\top W_\mathrm{src}`, which is itself
    orthogonal.

    Restriction maps are parameterised via:

    * **Cayley** (default): skew-symmetric matrix :math:`\to` orthogonal via
      Cayley transform.
    * **fasth**: Householder reflections.
    * **attention**: Cayley transform of an attention-derived skew-symmetric matrix.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        clamp_val: float = 10.0,
        use_attention: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        use_edge_weights: bool = False,
    ):
        r"""Initializes an orthogonal SheafAdjacency layer.

        The map generator outputs parameters for either: (1) entries of a
        skew-symmetric matrix (cayley), (2) Householder vectors (fasth),
        or (3) attention-based mappings.  All parameterisations produce
        orthogonal :math:`d \\times d` restriction maps.

        Args:
            stalk_dim: Stalk dimension and orthogonal restriction-map matrix size.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: If ``True``, self-loops augment the degree used for
                normalisation. Defaults to ``True``.
            clamp_val: Maximum absolute value for clamping Cayley-transform
                parameters. Defaults to 10.0.
            use_attention: If ``True``, uses the attention-based Cayley
                initialisation. Defaults to ``False``.
            orth_strategy: ``"cayley"`` or ``"fasth"``. Defaults to ``"cayley"``.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to
                ``False``.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
            use_edge_weights: Learned symmetric per-edge scalars rescaling
                orthogonal maps (reference edge_weights). Defaults to False.
        """
        if orth_strategy not in ORTH_STRATEGIES:
            raise ValueError(
                f"orth_strategy must be one of {ORTH_STRATEGIES}, got {orth_strategy!r}"
            )
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.clamp_val = clamp_val
        self.use_attention = use_attention
        self.orth_strategy = orth_strategy
        self.use_edge_weights = use_edge_weights
        if use_edge_weights:
            # Reference EdgeWeightLearner: bias-free scalar head, sigmoid at use.
            self.weight_generator = nn.Linear(2 * self.context_dim, 1, bias=False)

        num_params = (
            stalk_dim * stalk_dim
            if use_attention
            else orth_param_count(orth_strategy, stalk_dim)
        )

        self.map_generator = nn.Sequential(
            nn.Linear(self._generator_in_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Orthogonal degree normalisation; a 2-D self_map carries the
        per-channel weighted diagonal from use_edge_weights.
        """
        self_diag = self_map if self_map is not None and self_map.dim() == 2 else None
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes, self_diag)

    def get_map_products(self, x_feat, edge_index):
        """Compute orthogonal cross-map W_dst^T W_src and identity
        self-map for all edges.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(identity, cross_map)`` where both are [E, d, d].
            ``self_map`` is the identity because W^T W = I.
        """
        params_dst, params_src = self._bidirectional_input(x_feat, edge_index)

        if self.use_attention:
            W_dst = attention_cayley(params_dst, self.d, x_feat.device, x_feat.dtype)
            W_src = attention_cayley(params_src, self.d, x_feat.device, x_feat.dtype)
        else:
            W_dst = build_orthogonal(
                params_dst, self.d, self.orth_strategy, self.clamp_val
            )
            W_src = build_orthogonal(
                params_src, self.d, self.orth_strategy, self.clamp_val
            )

        # W_dst^T W_dst = I (orthogonality), so self_map = I.
        cross_map = torch.matmul(W_dst.transpose(-2, -1), W_src)  # [E, d, d]
        E, d, _ = cross_map.shape

        if self.use_edge_weights:
            # Symmetric per-edge scalar w = sigmoid(w_uv) * sigmoid(w_vu); the
            # products scale by w^2 while fixed lp/hp channels keep unit weight.
            w_a, w_b = self._bidirectional_pair(
                self.weight_generator, x_feat, edge_index
            )
            w2 = (torch.sigmoid(w_a) * torch.sigmoid(w_b)).squeeze(-1) ** 2
            _, cross_map = self._append_fixed_block(
                cross_map, w2[:, None, None] * cross_map
            )
            # 2-D self carries the per-channel weighted diagonal to _apply_norm.
            self_diag = cross_map.new_ones(E, self.stalk_dim)
            self_diag[:, :d] = w2.unsqueeze(-1)
            return self_diag, cross_map

        identity = torch.eye(d, device=cross_map.device, dtype=cross_map.dtype).expand(
            E, d, d
        )
        return self._append_fixed_block(identity, cross_map)

    def forward(
        self, x_feat: torch.Tensor, x_stalk: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Applies one orthogonal SheafAdjacency step.

        Overrides the base ``forward()`` because ``apply_orthogonal_norm`` takes
        only ``cross_map`` (``self_map`` is ``None`` for the orthogonal path).
        """
        z = self._apply_stalk_transform(x_stalk)
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        src_idx, _dst_idx = edge_index

        # self_map is the identity, or the per-channel weighted diagonal (2-D)
        # when use_edge_weights is on; the norm needs it for the degree.
        self_map, cross_map = self.get_map_products(x_feat, edge_index)
        _, norm_cross = self._apply_norm(self_map, cross_map, edge_index, num_nodes)

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


class LowRankSheafAdjConv(BaseSheafAdjConv):
    r"""SheafAdjacency layer with low-rank restriction maps.

    Parameterises each restriction map as :math:`F = A B^\top` where
    :math:`A, B \in \mathbb{R}^{d \times r}`, bounding the effective rank to
    at most ``rank``.  This is cheaper than
    :class:`GeneralSheafAdjConv` (:math:`2dr` vs :math:`d^2` parameters per map)
    while keeping the full :math:`d`-dimensional stalk.

    The map products reduce to

    .. math::

        F_\mathrm{dst}^\top F_\mathrm{dst}
            &= B_\mathrm{dst}\,(A_\mathrm{dst}^\top A_\mathrm{dst})\,B_\mathrm{dst}^\top
            \in \mathbb{R}^{d \times d}, \\
        F_\mathrm{dst}^\top F_\mathrm{src}
            &= B_\mathrm{dst}\,(A_\mathrm{dst}^\top A_\mathrm{src})\,B_\mathrm{src}^\top
            \in \mathbb{R}^{d \times d}.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        rank: int = 1,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initializes a low-rank SheafAdjacency layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension f inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
            rank: Rank r of each restriction map F = A @ B^T. Must be positive.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to False.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim,
            add_self_loops,
            add_eps,
            add_lp,
            add_hp,
            sparse_learner,
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.map_generator = nn.Sequential(
            nn.Linear(self._generator_in_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stalk_dim * rank),
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Apply low-rank degree normalisation to self- and cross-map products."""
        return apply_low_rank_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
            learned_dim=self.d,
        )

    def get_map_products(self, x_feat, edge_index):
        """Compute low-rank map products F_dst^T F_dst and F_dst^T F_src for all edges.

        Each map F = left @ right^T so F^T F = right (left^T left) right^T.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d, d].
        """
        raw_dst, raw_src = self._bidirectional_input(x_feat, edge_index)

        # Reshape into [E, d, 2r], then split into left [E, d, r] and right [E, d, r].
        # Restriction map F = left @ right^T, so F^T = right @ left^T.
        left_dst, right_dst = raw_dst.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)
        left_src, right_src = raw_src.view(-1, self.d, 2 * self.rank).chunk(2, dim=-1)

        gram_dst_dst = torch.matmul(left_dst.transpose(-2, -1), left_dst)  # [E, r, r]
        gram_dst_src = torch.matmul(left_dst.transpose(-2, -1), left_src)  # [E, r, r]

        self_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_dst, right_dst.transpose(-2, -1))
        )
        cross_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_src, right_src.transpose(-2, -1))
        )
        return self._append_fixed_block(self_map, cross_map)


__all__ = [
    "BaseSheafAdjConv",
    "DiagonalSheafAdjConv",
    "GeneralSheafAdjConv",
    "OrthogonalSheafAdjConv",
    "LowRankSheafAdjConv",
]
