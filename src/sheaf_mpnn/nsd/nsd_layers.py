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


class BaseNSDConv(BaseSheafConv):
    r"""Base class for NSD convolutions (MPSNN formulation).

    The class computes the full sheaf Laplacian action inside ``message()``
    without a separate scatter for the diagonal term.  Each subclass implements
    ``get_map_products()``, which precomputes the composed restriction-map products

    .. math::

        \texttt{self\_map} = F_\mathrm{dst}^\top F_\mathrm{dst}, \qquad
        \texttt{cross\_map} = F_\mathrm{dst}^\top F_\mathrm{src}

    per edge before the message loop, reducing ``message()`` to two matmuls
    instead of three.
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
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initializes the shared NSD convolution parameters.

        Args:
            stalk_dim: Number of LEARNED stalk dimensions (reference ``d``).
                With ``add_lp``/``add_hp`` the full stalk handled by the layer
                is ``final_d = stalk_dim + add_lp + add_hp`` and node states
                have shape ``[final_d, in_channels]``.
            in_channels: Feature dimension inside each stalk channel (f).
            hidden_dim: Unused by the reference-style linear map generator;
                kept for API compatibility.
            alpha: Initial residual diffusion step size.
            context_dim: Width of each node context vector ``x_feat``.
            add_self_loops: Whether to add self-loops for degree normalization.
                Defaults to ``False``; the normalization is already
                (deg+1)-augmented, so self-loops would augment twice.
            add_eps: If ``True``, adds a learnable per-stalk rescaling
                ``eps`` in R^d (Remark 4, Appendix A.1). The skip term
                ``x_stalk`` is replaced by ``(1 + eps) * x_stalk``, where eps
                is broadcast across the feature channel dimension. Initialized
                to zero so default behaviour is the standard NSD update.
            dropout: Dropout probability applied to the diffusion input (the
                skip path stays un-dropped), matching the reference per-layer
                dropout before the left/right weights.
            add_lp: Appends one fixed stalk channel whose Laplacian block is
                the signless graph Laplacian D+A (sharpening), as in the
                reference ``--add_lp``.
            add_hp: Appends one fixed stalk channel whose Laplacian block is
                the standard graph Laplacian D-A (smoothing), as in the
                reference ``--add_hp``.
            sparse_learner: Uses the reference LocalConcatSheafLearnerVariant
                input reduction: the concatenated pair is viewed as
                ``[final_d, 2f]`` and summed over the stalk axis, shrinking
                the learner input from ``2*final_d*f`` to ``2f``.
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
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, x_feat: torch.Tensor, x_stalk: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Applies one NSD diffusion step to lifted node features.

        Args:
            x_feat (torch.Tensor): Node context features [num_nodes, context_dim].
            x_stalk (torch.Tensor): Lifted node features [num_nodes, d, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Updated stalk features [num_nodes, d, in_channels].
        """
        # Dropout hits only the diffusion input; the skip term below keeps the
        # un-dropped x_stalk, as in the reference update x0 - ELU(L drop(x) W).
        z = self._apply_stalk_transform(self.dropout_layer(x_stalk))
        num_nodes = x_stalk.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        src_idx, dst_idx = edge_index

        self_map, cross_map = self.get_map_products(x_feat, edge_index)
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        z_src, z_dst = z[src_idx], z[dst_idx]
        laplacian_out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )

        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip - self.alpha * self.sigma(laplacian_out)

    @abstractmethod
    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""

    def _build_map_generator(self, out_dim: int, bounded: bool = True) -> nn.Sequential:
        """Reference-style sheaf learner: single bias-free linear, optionally
        tanh-bounded so every map entry lies in (-1, 1).
        """
        layers: list[nn.Module] = [
            nn.Linear(self._generator_in_dim(), out_dim, bias=False)
        ]
        if bounded:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)


class DiagonalNSDConv(BaseNSDConv):
    """Diagonal NSD convolution layer."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        dropout: float = 0.0,
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
            dropout,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.map_generator = self._build_map_generator(stalk_dim)
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

        # Element-wise products [E, d] instead of [E, d, d] diagonal matrices,
        # extended with the fixed lp/hp channels when enabled.
        return self._append_fixed_diag(w_dst**2, w_dst * w_src)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        """Builds diagonal Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Normalised w_dst**2 per edge [E, d].
            cross_map: Normalised w_dst*w_src per edge [E, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        # self_map = w_dst^2 [E, d], cross_map = w_dst * w_src [E, d].
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src


class GeneralNSDConv(BaseNSDConv):
    """Generalized NSD convolution layer."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        add_eps: bool = True,
        dropout: float = 0.0,
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
            dropout,
            add_lp,
            add_hp,
            sparse_learner,
        )
        self.use_attention = use_attention
        # Attention maps go through a softmax, so they stay unbounded (reference
        # AttentionSheafLearner has no tanh); plain general maps are tanh-bounded.
        self.map_generator = self._build_map_generator(
            stalk_dim * stalk_dim, bounded=not use_attention
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


class OrthogonalNSDConv(BaseNSDConv):
    """Orthogonal NSD convolution layer."""

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
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        use_edge_weights: bool = False,
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
                used for normalization. Defaults to ``False``.
            clamp_val (float, optional): Maximum absolute value for clamping
                Cayley-transform parameters. Defaults to 10.0.
            use_attention (bool, optional): If ``True``, uses the attention-based
                Cayley initialization from main. Defaults to ``False``.
            orth_strategy (str, optional): "cayley" or "fasth". Defaults to "cayley".
            add_eps (bool, optional): If ``True``, adds a learnable per-stalk rescaling
                ``eps in R^d`` to the skip term (Remark 4). Defaults to ``True``.
            dropout (float, optional): Dropout probability on the diffusion
                input. Defaults to 0.0.
            use_edge_weights (bool, optional): Learns a symmetric per-edge
                scalar (reference EdgeWeightLearner) that rescales the
                orthogonal transport maps and replaces the combinatorial
                degree with the weighted one. Defaults to ``False``.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
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
            dropout,
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

        # Tanh-bounded params feed the orthogonal transform, as in the reference
        # (sheaf_act tanh before Orthogonal); attention params stay unbounded.
        self.map_generator = self._build_map_generator(
            num_params, bounded=not use_attention
        )
        self.reset_parameters()

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Orthogonal degree normalisation; a 2-D self_map carries the
        per-channel weighted diagonal from use_edge_weights.
        """
        self_diag = self_map if self_map.dim() == 2 else None
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes, self_diag)

    def get_map_products(self, x_feat, edge_index):
        """Compute orthogonal cross-map W_dst^T W_src and identity
        self-map for all edges.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(identity, cross_map)`` where ``identity`` is [E, d, d] and
            ``cross_map`` is [E, d, d].  ``self_map`` is the identity because W^T W = I.
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
            # Per-channel diagonal: w^2 on learned dims, 1 on fixed channels;
            # 2-D shape signals the weighted-degree path to _apply_norm.
            self_diag = cross_map.new_ones(E, self.stalk_dim)
            self_diag[:, :d] = w2.unsqueeze(-1)
            return self_diag, cross_map

        identity = torch.eye(d, device=cross_map.device, dtype=cross_map.dtype).expand(
            E, d, d
        )
        # Fixed lp/hp channels extend both to [E, final_d, final_d]; the self
        # block stays the identity because the fixed diagonal entries are 1.
        return self._append_fixed_block(identity, cross_map)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        """Builds orthogonal Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Scalar degree inverse D^{-1}_dst broadcast to [E, 1, 1].
            cross_map: Normalised D^{-1/2}_dst W_dst^T W_src
                D^{-1/2}_src per edge [E, d, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        # self_map = D^{-1}_dst [E,1,1]; cross_map = D^{-1/2}_dst W^T W D^{-1/2}_src.
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class LowRankNSDConv(BaseNSDConv):
    """Low-rank NSD convolution layer.

    Parameterizes each restriction map as F = A @ B^T where A, B in R^{dxr},
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
        add_self_loops: bool = False,
        rank: int = 1,
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initializes a low-rank NSD convolution layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            alpha: Initial learnable diffusion step size. Defaults to 1.0.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
                Defaults to ``False``; the normalization is already
                (deg+1)-augmented, so self-loops would augment twice.
            rank: Rank of each restriction map F = A @ B^T. Must be positive.
            add_eps: Learnable per-stalk skip rescaling (Remark 4). Defaults to True.
            dropout: Dropout probability on the diffusion input. Defaults to 0.0.
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
            dropout,
            add_lp,
            add_hp,
            sparse_learner,
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.map_generator = self._build_map_generator(2 * stalk_dim * rank)
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
    "BaseNSDConv",
    "DiagonalNSDConv",
    "GeneralNSDConv",
    "OrthogonalNSDConv",
    "LowRankNSDConv",
]
