# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class BaseSheafConv(MessagePassing):
    """Shared base for all NSD and NSP sheaf convolution layers.

    Factors out the parameterization and utilities that are identical across
    every variant in both model families:

    * ``W1`` / ``W2`` -- bilateral stalk transforms (left d*d, right f*f).
    * ``sigma`` -- activation function (Tanh).
    * ``reset_parameters()`` -- Xavier init for W1, W2, and any map_generator.
    * ``_apply_stalk_transform(x)`` -- computes ``W1 @ x @ W2``.
    * ``_apply_norm(...)`` -- abstract; each concrete subclass delegates to the
      appropriate ``apply_*_norm`` function from ``sheaf_mpnn.utils``.

    Concrete subclasses must implement:
        ``get_map_products(x_feat, edge_index) -> (self_map, cross_map)``
        ``_apply_norm(self_map, cross_map, edge_index, num_nodes)``
        ``forward(x_feat, x_stalk, edge_index) -> updated stalk``
        ``message(...)``
    """

    map_generator: nn.Module
    weight_generator: nn.Linear

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)
        # stalk_dim counts the LEARNED dims (self.d); the fixed lp/hp channels
        # extend the full stalk handled by the layer to self.stalk_dim.
        self.d = stalk_dim
        self.stalk_dim = stalk_dim + int(add_lp) + int(add_hp)
        self.add_lp = add_lp
        self.add_hp = add_hp
        self.sparse_learner = sparse_learner
        self.in_channels = in_channels  # 'f' (feature dimension per stalk entry)
        self.context_dim = (
            context_dim if context_dim is not None else (self.stalk_dim * in_channels)
        )
        self.add_self_loops = add_self_loops

        self.W1 = nn.Parameter(torch.empty(self.stalk_dim, self.stalk_dim))
        self.W2 = nn.Parameter(torch.empty(in_channels, in_channels))
        # ELU on the Laplacian output (reference use_act); tanh would cap every
        # update entry at +/-1 and saturate on large diffusion values.
        self.sigma: nn.Module = nn.ELU()

    def reset_parameters(self):
        # Reference init: identity left transform (no stalk mixing at init),
        # orthogonal right transform (norm-preserving channel mixing).
        nn.init.eye_(self.W1)
        nn.init.orthogonal_(self.W2)
        # Covers both the singular map_generator (NSD) and the
        # map_generators ModuleList (SheafAttnConv).
        generators: list[nn.Sequential] = []
        if hasattr(self, "map_generator"):
            gen = self.map_generator
            if isinstance(gen, nn.Sequential):
                generators.append(gen)
        if hasattr(self, "map_generators"):
            map_gens = self.map_generators
            if isinstance(map_gens, nn.ModuleList):
                for gen in map_gens:
                    if isinstance(gen, nn.Sequential):
                        generators.append(gen)
        for gen in generators:
            for m in gen:
                if isinstance(m, nn.Linear):
                    # Default init (reference parity): a near-zero warm start is
                    # defeated by the degree normalization anyway.
                    m.reset_parameters()
        # Edge-weight learner (reference EdgeWeightLearner), when present.
        weight_gen = getattr(self, "weight_generator", None)
        if isinstance(weight_gen, nn.Linear):
            weight_gen.reset_parameters()

    def _apply_stalk_transform(self, x):
        """Applies bilateral stalk transform: W1 @ x @ W2."""
        return torch.matmul(torch.matmul(self.W1, x), self.W2)

    @abstractmethod
    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        r"""Normalizes restriction-map products by the sheaf degree :math:`D^{-1/2}`.

        Each concrete subclass calls the matching ``apply_*_norm`` utility from
        ``sheaf_mpnn.utils``:

        * Diagonal   -> ``apply_diagonal_norm``
        * Orthogonal -> ``apply_orthogonal_norm``
        * Low-rank   -> ``apply_low_rank_norm``
        * General    -> ``apply_general_norm``

        Returns:
            (norm_self, norm_cross): Normalized products ready for ``message()``.
        """
        raise NotImplementedError("Subclasses must implement _apply_norm.")

    def message(  # ty: ignore[invalid-method-override]
        self, z_dst, z_src, self_map, cross_map
    ):
        r"""Builds per-edge sheaf Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Normalized :math:`F_\mathrm{dst}^\top F_\mathrm{dst}`
                per edge [E, d, d].
            cross_map: Normalized :math:`F_\mathrm{dst}^\top F_\mathrm{src}`
                per edge [E, d, d].

        Returns:
            torch.Tensor: Per-edge messages [E, d, f].
        """
        return torch.matmul(self_map, z_dst) - torch.matmul(cross_map, z_src)

    def _generator_in_dim(self) -> int:
        """Learner input width; the sparse learner sums the stalk axis of the
        concatenated pair, shrinking it from 2*final_d*f to 2*f.
        """
        return 2 * self.in_channels if self.sparse_learner else 2 * self.context_dim

    def _bidirectional_pair(
        self,
        generator: nn.Module,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch both edge orientations into a single generator pass then split."""
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]
        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        if sparse:
            # Reference LocalConcatSheafLearnerVariant reduction (requires
            # context_dim == final_d * in_channels).
            inp = inp.view(inp.size(0), self.stalk_dim, 2 * self.in_channels)
            inp = inp.sum(dim=1)
        return generator(inp).chunk(2, dim=0)

    def _bidirectional_input(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bidirectional_pair(
            self.map_generator, x_feat, edge_index, self.sparse_learner
        )

    def _append_fixed_diag(
        self, self_map: torch.Tensor, cross_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends the add_lp/add_hp fixed channels to [E, d] diagonal products.

        The per-edge neighbour contribution is +cross in every family, so
        cross = -1 yields the sharpening signless-Laplacian channel (lp) and
        cross = +1 the smoothing standard-Laplacian channel (hp); self = 1
        accumulates the plain degree.
        """
        if not (self.add_lp or self.add_hp):
            return self_map, cross_map
        ones = self_map.new_ones(self_map.size(0), 1)
        self_cols, cross_cols = [self_map], [cross_map]
        if self.add_lp:
            self_cols.append(ones)
            cross_cols.append(-ones)
        if self.add_hp:
            self_cols.append(ones)
            cross_cols.append(ones)
        return torch.cat(self_cols, dim=1), torch.cat(cross_cols, dim=1)

    def _append_fixed_block(
        self, self_map: torch.Tensor, cross_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds [E, d, d] products into [E, final_d, final_d] block-diagonal
        form with the add_lp/add_hp fixed channels on the trailing diagonal
        (same sign convention as _append_fixed_diag).
        """
        if not (self.add_lp or self.add_hp):
            return self_map, cross_map
        num_edges, d = self_map.size(0), self.d
        full_self = self_map.new_zeros(num_edges, self.stalk_dim, self.stalk_dim)
        full_cross = cross_map.new_zeros(num_edges, self.stalk_dim, self.stalk_dim)
        full_self[:, :d, :d] = self_map
        full_cross[:, :d, :d] = cross_map
        idx = d
        if self.add_lp:
            full_self[:, idx, idx] = 1.0
            full_cross[:, idx, idx] = -1.0
            idx += 1
        if self.add_hp:
            full_self[:, idx, idx] = 1.0
            full_cross[:, idx, idx] = 1.0
        return full_self, full_cross


class BaseDirectedSheafConv(MessagePassing):
    """Base for directed sheaf convolutions with per-direction bilateral transforms.

    Inherits directly from MessagePassing with directional pairs
    (W1_fwd, W2_fwd) and (W1_bwd, W2_bwd). Features are first propagated
    through normalised restriction maps, then mixed by the W transforms —
    matching the DirGCN structure: W(sheaf_adj @ x).
    Also provides _bidirectional_input for subclasses that use a map_generator.
    """

    map_generator: nn.Module

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels
        self.context_dim = (
            context_dim if context_dim is not None else (stalk_dim * in_channels)
        )
        self.add_self_loops = add_self_loops
        self.sigma = nn.Tanh()
        self.W1_fwd = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2_fwd = nn.Parameter(torch.empty(in_channels, in_channels))
        self.W1_bwd = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2_bwd = nn.Parameter(torch.empty(in_channels, in_channels))

    def reset_parameters(self):
        for W in [self.W1_fwd, self.W2_fwd, self.W1_bwd, self.W2_bwd]:
            nn.init.xavier_uniform_(W)
        if hasattr(self, "map_generator") and isinstance(
            self.map_generator, nn.Sequential
        ):
            for m in self.map_generator:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _apply_stalk_transform_fwd(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.W1_fwd, x), self.W2_fwd)

    def _apply_stalk_transform_bwd(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.W1_bwd, x), self.W2_bwd)

    def _bidirectional_input(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_idx, dst_idx = edge_index
        x_dst, x_src = x_feat[dst_idx], x_feat[src_idx]
        inp = torch.cat(
            [
                torch.cat([x_dst, x_src], dim=-1),
                torch.cat([x_src, x_dst], dim=-1),
            ],
            dim=0,
        )
        return self.map_generator(inp).chunk(2, dim=0)

    def message(  # ty: ignore[invalid-method-override]
        self, z_src: torch.Tensor, F: torch.Tensor
    ) -> torch.Tensor:
        """Applies the normalised restriction-map cross-product to source features.

        Args:
            z_src: Source stalk features [E, d, f].
            F:     Normalised cross-product map [E, d, d].

        Returns:
            torch.Tensor: Per-edge messages [E, d, f].
        """
        return torch.matmul(F, z_src)


__all__ = ["BaseSheafConv", "BaseDirectedSheafConv"]
