# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import (
    apply_diagonal_norm,
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
    attention_cayley,
    build_orthogonal,
    orth_param_count,
)


class BaseNSPConv(BaseSheafConv):
    """Base class for NSP convolutions.

    Implements the second-order NSP recurrence using the augmented state
    ``x_stalk = [x_curr; x_hist]``. The Laplacian is computed on ``x_curr`` only,
    then ``forward()`` returns ``[2 * x_curr - x_hist - sigma(laplacian); x_curr]``.
    See ``README.md`` for the paper-aligned state-augmentation derivation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initialize the layer.

        Args:
            stalk_dim: Learned stalk dimension d; add_lp/add_hp append fixed
                channels so the full stalk (and half the augmented state) is
                final_d = d + add_lp + add_hp.
            in_channels: Feature dimension f.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            context_dim: Width of each node context vector. Defaults to d*f.
            add_self_loops: Whether to add self-loops for degree normalization.
            add_eps: If ``True``, adds learnable per-stalk rescalings ``eps_curr``
                and ``eps_hist`` in R^d (Remark 6, Appendix A.1). The recurrence
                becomes ``(2+eps_curr)*x_curr - (1+eps_hist)*x_hist - sigma(lap)``.
                Both are initialized to zero so default behaviour is standard NSP.
            add_lp: Appends one fixed stalk channel diffusing with the signless
                Laplacian D+A (reference ``--add_lp``).
            add_hp: Appends one fixed stalk channel diffusing with the standard
                Laplacian D-A (reference ``--add_hp``).
            sparse_learner: Stalk-summed map-generator input, shrinking the learner
                from 2*final_d*hidden to 2*hidden inputs.
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
        self.add_eps = add_eps
        if add_eps:
            self.eps_curr = nn.Parameter(torch.zeros(self.stalk_dim))
            self.eps_hist = nn.Parameter(torch.zeros(self.stalk_dim))

    def forward(self, x_feat, x_stalk, edge_index):
        """Applies one NSP propagation step on the augmented state.

        Args:
            x_feat (Tensor): [N, context_dim] flattened current stalk, used as
                context for the map-generator MLP.  Callers extract this as
                x_stalk[:, :d, :].reshape(N, -1).
            x_stalk (Tensor): [N, 2d, f] augmented node stalks.  First d rows
                hold x_curr = x_u^(t); last d rows hold x_hist = x_u^(t-1).
            edge_index (LongTensor): [2, E] graph connectivity.

        Returns:
            Tensor: [N, 2d, f] updated augmented stalks.
        """
        x_curr = x_stalk[:, : self.stalk_dim, :]  # [N, d, f]
        x_hist = x_stalk[:, self.stalk_dim :, :]  # [N, d, f]

        z = self._apply_stalk_transform(x_curr)
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

        if self.add_eps:
            curr_scale = 2 + torch.tanh(self.eps_curr)[None, :, None]
            hist_scale = 1 + torch.tanh(self.eps_hist)[None, :, None]
            new_curr = (
                curr_scale * x_curr - hist_scale * x_hist - self.sigma(laplacian_out)
            )
        else:
            new_curr = 2 * x_curr - x_hist - self.sigma(laplacian_out)
        return torch.cat([new_curr, x_curr], dim=1)  # [N, 2d, f]

    @abstractmethod
    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute self_map and cross_map restriction-map products per edge."""


class DiagonalNSPConv(BaseNSPConv):
    r"""NSP layer with diagonal restriction maps.

    Each restriction map is a diagonal :math:`d \times d` matrix represented by
    a vector :math:`w` of length :math:`d`, making
    :math:`F^\top F = \mathrm{diag}(w^2)` and
    :math:`F_\mathrm{dst}^\top F_\mathrm{src} =
    \mathrm{diag}(w_\mathrm{dst} \cdot w_\mathrm{src})`.
    ``message()`` uses element-wise multiply instead of matmul:
    :math:`O(E \cdot d \cdot f)` instead of :math:`O(E \cdot d^2 \cdot f)`.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
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
        return apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index):
        """Compute diagonal map products w_dst**2 and w_dst*w_src for all edges.

        Args:
            x_feat: Node context features [N, context_dim].
            edge_index: Graph connectivity [2, E].

        Returns:
            tuple: ``(self_map, cross_map)`` both [E, d] -- element-wise products.
        """
        w_dst, w_src = self._bidirectional_input(x_feat, edge_index)

        return self._append_fixed_diag(w_dst**2, w_dst * w_src)

    def message(self, z_dst, z_src, self_map, cross_map):
        """Builds diagonal Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Normalised w_dst**2 per edge [E, d].
            cross_map: Normalised w_dst*w_src per edge [E, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src


class GeneralNSPConv(BaseNSPConv):
    """NSP layer with general restriction maps.

    The map-generator MLP produces one full square matrix per edge orientation,
    giving the most expressive restriction maps among the NSP variants.

    When use_attention=True, the generated maps are built from an attention-style
    softmax transform that biases them toward near-identity maps at initialisation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
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
            phi_dst = eye - torch.softmax(phi_dst, dim=-1)
            phi_src = eye - torch.softmax(phi_src, dim=-1)

        self_map = torch.matmul(phi_dst.transpose(-2, -1), phi_dst)  # [E, d, d]
        cross_map = torch.matmul(phi_dst.transpose(-2, -1), phi_src)  # [E, d, d]
        return self._append_fixed_block(self_map, cross_map)


class OrthogonalNSPConv(BaseNSPConv):
    r"""NSP layer with orthogonal restriction maps.

    Supports three parameterisations, all producing orthogonal :math:`d \times d` maps:

    * ``cayley`` (default): maps the :math:`(d^2 - d)/2` upper-triangle entries of a
      skew-symmetric :math:`A` to :math:`W = (I - A)^{-1}(I + A)` via the Cayley
      transform.
    * ``fasth``: uses a batched FastH Householder decomposition of :math:`d^2` raw
      parameters.
    * attention (``use_attention=True``): builds :math:`A` from the skew-symmetric
      part of :math:`I - \mathrm{softmax}(\mathrm{raw})` before applying the Cayley
      transform.

    Because :math:`W_v^\top W_v = I` for all variants, the diagonal Laplacian term
    simplifies to ``z_dst``, so ``forward()`` and ``message()`` skip the
    ``t_diag`` computation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        clamp_val: float = 10.0,
        use_attention: bool = False,
        orth_strategy: str = "cayley",
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        use_edge_weights: bool = False,
    ):
        """Initializes an orthogonal NSP convolution layer.

        Args:
            stalk_dim (int): Stalk dimension and orthogonal restriction-map matrix size.
            in_channels (int): Feature dimension inside each stalk channel.
            hidden_dim (int): Hidden width of the restriction-map generator MLP.
            context_dim (int, optional): Width of ``x_feat``. Defaults to
                ``d * in_channels`` when omitted.
            add_self_loops (bool, optional): If ``True``, self-loops augment the
                degree used for normalization. Defaults to ``True``.
            clamp_val (float, optional): Maximum absolute value for clamping
                Cayley-transform parameters. Defaults to 10.0.
            use_attention (bool, optional): If ``True``, uses the attention-based
                Cayley initialization. Defaults to ``False``.
            orth_strategy (str, optional): "cayley" or "fasth". Defaults to "cayley".
            add_eps: Learnable per-stalk skip rescaling (Remark 6). Defaults to False.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
            use_edge_weights: Learned symmetric per-edge scalars rescaling
                orthogonal maps (reference edge_weights). Defaults to False.
        """
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
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

    def forward(self, x_feat, x_stalk, edge_index):
        """Applies one orthogonal NSP step (self_map = I, skips diagonal self-term).

        Args:
            x_feat: [N, context_dim] flattened current stalk for the map-generator MLP.
            x_stalk: [N, 2d, f] augmented stalks (curr||hist).
            edge_index: [2, E] graph connectivity.

        Returns:
            torch.Tensor: [N, 2d, f] updated augmented stalks.
        """
        x_curr = x_stalk[:, : self.stalk_dim, :]
        x_hist = x_stalk[:, self.stalk_dim :, :]
        z = self._apply_stalk_transform(x_curr)
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
        # self_map = D^{-1}_dst [E,1,1]; cross_map = D^{-1/2}_dst W^T W D^{-1/2}_src
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class LowRankNSPConv(BaseNSPConv):
    r"""NSP layer with low-rank restriction maps.

    Parameterizes each restriction map as
    :math:`F = \mathrm{left} \cdot \mathrm{right}^\top`
    where :math:`\mathrm{left}, \mathrm{right} \in \mathbb{R}^{d \times r}`,
    bounding the effective rank to at most ``rank``.  This gives cheaper
    parameterization than
    ``GeneralNSPConv`` (:math:`2dr` vs :math:`d^2` params per map) while keeping
    the full :math:`d`-dimensional stalk.

    The Laplacian products reduce to:

    .. math::

        F_\mathrm{dst}^\top F_\mathrm{dst}
            &= R_\mathrm{dst}
               \bigl(L_\mathrm{dst}^\top L_\mathrm{dst}\bigr)
               R_\mathrm{dst}^\top
               \in \mathbb{R}^{d \times d} \\
        F_\mathrm{dst}^\top F_\mathrm{src}
            &= R_\mathrm{dst}
               \bigl(L_\mathrm{dst}^\top L_\mathrm{src}\bigr)
               R_\mathrm{src}^\top
               \in \mathbb{R}^{d \times d}
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        rank: int = 1,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        """Initializes a low-rank NSP convolution layer.

        Args:
            stalk_dim: Stalk dimension.
            in_channels: Feature dimension inside each stalk channel.
            hidden_dim: Hidden width of the restriction-map generator MLP.
            context_dim: Width of ``x_feat``. Defaults to ``d * in_channels``.
            add_self_loops: Whether to add self-loops for degree normalization.
            rank: Rank of each restriction map F = left @ right^T. Must be positive.
            add_eps: Learnable per-stalk skip rescaling (Remark 6). Defaults to False.
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


class ConnectionLaplacianNSPConv(BaseNSPConv):
    """NSP convolution with connection-Laplacian restriction maps.

    Restriction maps are derived from input node features via local PCA and
    SVD-based orthogonal alignment — there is no learned map-generator MLP.
    Because F^T F = I the normalisation and message logic are identical to
    the orthogonal variant.

    The caller is responsible for passing precomputed per-edge orthogonal
    transport maps as the ``maps`` argument to ``forward`` / ``get_map_products``.
    See ``ConnectionLaplacianNSPModel`` for the model that handles this.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        add_self_loops: bool = False,
        add_eps: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim=1,
            add_self_loops=add_self_loops,
            add_eps=add_eps,
            add_lp=add_lp,
            add_hp=add_hp,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Orthogonal degree normalisation; a 2-D self_map carries the
        per-channel weighted diagonal from use_edge_weights.
        """
        self_diag = self_map if self_map is not None and self_map.dim() == 2 else None
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes, self_diag)

    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cross-map from precomputed connection-Laplacian transport maps.

        Args:
            x_feat: Per-edge orthogonal transport maps ``[E, d, d]`` as returned by
                ``_orthogonal_alignment``.  For edge ``(src, dst)`` the map is
                ``O_src^T O_dst``; transposing gives ``F_dst^T F_src = O_dst^T O_src``.
            edge_index: Graph connectivity ``[2, E]`` (unused; kept for API parity).

        Returns:
            ``(identity, cross_map)`` — self_map is the identity ``[E, d, d]``
            because ``F^T F = I`` for orthogonal transport maps.
        """
        cross_map = x_feat.transpose(-2, -1)
        E, d, _ = cross_map.shape
        identity = torch.eye(d, device=cross_map.device, dtype=cross_map.dtype).expand(
            E, d, d
        )
        return self._append_fixed_block(identity, cross_map)

    def message(self, z_dst, z_src, self_map, cross_map):
        """Builds connection-Laplacian messages using precomputed transport maps.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Scalar degree inverse D^{-1}_dst broadcast to [E, 1, 1].
            cross_map: Normalised D^{-1/2}_dst O_dst^T O_src
                D^{-1/2}_src per edge [E, d, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        # self_map = D^{-1}_dst [E,1,1]; cross_map = D^{-1/2}_dst O^T O D^{-1/2}_src.
        return self_map * z_dst - torch.matmul(cross_map, z_src)


__all__ = [
    "BaseNSPConv",
    "ConnectionLaplacianNSPConv",
    "DiagonalNSPConv",
    "GeneralNSPConv",
    "OrthogonalNSPConv",
    "LowRankNSPConv",
]
