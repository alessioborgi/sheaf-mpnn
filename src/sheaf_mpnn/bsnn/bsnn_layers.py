# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Bayesian NSD convolution layers (MPSNN formulation).

BSNN layers share the diffusion update of the deterministic NSD layers

.. math::

    X_{t+1} = (1 + \tanh\varepsilon)\,X_t
              - \alpha\,\sigma\big(\Delta_{\mathcal{F}^{(t)}}
              (I_n \otimes W_1) X_t W_2\big),

but the restriction maps :math:`\mathcal{F}^{(t)}` are **sampled** from a
variational distribution owned by the model rather than produced by a
per-layer map generator. Each layer therefore has no ``map_generator``; the
sampled maps arrive through the ``x_feat`` slot of ``forward`` as a stacked
pair ``[2, E, ...]`` of destination and source maps per directed edge (the
same trick ``ConnectionLaplacianNSDConv`` uses for its precomputed transport
maps).

The model samples one i.i.d. sheaf per layer from a posterior parameterized
once per forward pass, matching Algorithm 1 of Gillespie et al. (2025).
"""

from __future__ import annotations

import torch

from sheaf_mpnn.nsd.nsd_layers import BaseNSDConv
from sheaf_mpnn.utils import (
    apply_diagonal_norm,
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
)


class BaseBSNNConv(BaseNSDConv):
    r"""Base class for Bayesian NSD convolutions.

    Identical to ``BaseNSDConv`` except that no map generator is created:
    ``get_map_products`` consumes externally sampled restriction maps. The
    ``x_feat`` argument of ``forward`` carries the stacked sampled maps
    ``[2, E, ...]`` where index 0 holds the destination-node maps
    :math:`F_{\mathrm{dst} \trianglelefteq e}` and index 1 the source-node
    maps :math:`F_{\mathrm{src} \trianglelefteq e}` per directed edge.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
    ):
        if add_self_loops:
            raise ValueError(
                "BSNN layers receive per-edge sampled maps; adding self-loops "
                "inside the layer would desynchronize maps and edges."
            )
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=1,
            add_self_loops=False,
            add_eps=add_eps,
            dropout=dropout,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=False,
        )
        self.reset_parameters()


class DiagonalBSNNConv(BaseBSNNConv):
    """Bayesian NSD convolution with sampled diagonal restriction maps."""

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Apply diagonal degree normalisation to self- and cross-map products."""
        return apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index):
        """Compute diagonal products from sampled maps.

        Args:
            x_feat: Stacked sampled diagonal maps ``[2, E, d]``
                (``[0]`` = destination maps, ``[1]`` = source maps).
            edge_index: Graph connectivity ``[2, E]`` (unused; API parity).

        Returns:
            tuple: ``(self_map, cross_map)`` both ``[E, d]``.
        """
        w_dst, w_src = x_feat.unbind(0)
        return self._append_fixed_diag(w_dst**2, w_dst * w_src)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        """Builds diagonal Laplacian messages from normalised products.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Normalised ``w_dst**2`` per edge [E, d].
            cross_map: Normalised ``w_dst * w_src`` per edge [E, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src


class GeneralBSNNConv(BaseBSNNConv):
    """Bayesian NSD convolution with sampled general linear restriction maps."""

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
        """Compute Gram products from sampled general maps.

        Args:
            x_feat: Stacked sampled maps ``[2, E, d, d]``
                (``[0]`` = destination maps, ``[1]`` = source maps).
            edge_index: Graph connectivity ``[2, E]`` (unused; API parity).

        Returns:
            tuple: ``(self_map, cross_map)`` both ``[E, d, d]``.
        """
        maps_dst, maps_src = x_feat.unbind(0)
        self_map = torch.matmul(maps_dst.transpose(-2, -1), maps_dst)
        cross_map = torch.matmul(maps_dst.transpose(-2, -1), maps_src)
        return self._append_fixed_block(self_map, cross_map)


class LowRankBSNNConv(BaseBSNNConv):
    r"""Bayesian NSD convolution with sampled low-rank restriction maps.

    Each sampled map is :math:`F = AB^\top` with factors :math:`A, B \in
    \mathbb{R}^{d \times r}` drawn by ``LowRankVariationalSheafLearner``, so
    every sample has rank at most ``rank``. The maps arrive in factor form
    ``[2, E, d, 2r]`` (left factor then right factor along the last axis) and
    the Laplacian products are formed without materializing :math:`F`, as in
    ``LowRankNSDConv``:

    .. math::

        F_\mathrm{dst}^\top F_\mathrm{dst}
            &= B_\mathrm{dst} (A_\mathrm{dst}^\top A_\mathrm{dst})
               B_\mathrm{dst}^\top, \\
        F_\mathrm{dst}^\top F_\mathrm{src}
            &= B_\mathrm{dst} (A_\mathrm{dst}^\top A_\mathrm{src})
               B_\mathrm{src}^\top.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        rank: int = 1,
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            add_self_loops,
            add_eps,
            dropout,
            add_lp,
            add_hp,
        )
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank

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
        """Compute low-rank Gram products from sampled factors.

        Args:
            x_feat: Stacked sampled factors ``[2, E, d, 2r]``
                (``[0]`` = destination factors, ``[1]`` = source factors; the
                last axis holds the left factor ``A`` then the right factor
                ``B``).
            edge_index: Graph connectivity ``[2, E]`` (unused; API parity).

        Returns:
            tuple: ``(self_map, cross_map)`` both ``[E, d, d]``.
        """
        if x_feat.size(-1) != 2 * self.rank:
            raise ValueError(
                f"sampled factors must have trailing dimension {2 * self.rank}"
                f" (= 2 * rank), got {x_feat.size(-1)}"
            )
        factors_dst, factors_src = x_feat.unbind(0)
        left_dst, right_dst = factors_dst.chunk(2, dim=-1)  # [E, d, r] each
        left_src, right_src = factors_src.chunk(2, dim=-1)

        gram_dst_dst = torch.matmul(left_dst.transpose(-2, -1), left_dst)  # [E, r, r]
        gram_dst_src = torch.matmul(left_dst.transpose(-2, -1), left_src)  # [E, r, r]

        self_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_dst, right_dst.transpose(-2, -1))
        )
        cross_map = torch.matmul(
            right_dst, torch.matmul(gram_dst_src, right_src.transpose(-2, -1))
        )
        return self._append_fixed_block(self_map, cross_map)


class OrthogonalBSNNConv(BaseBSNNConv):
    """Bayesian NSD convolution with sampled special orthogonal maps."""

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        """Orthogonal degree normalisation (scalar degree; self_map is identity)."""
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def get_map_products(self, x_feat, edge_index):
        r"""Compute the orthogonal cross-product from sampled rotations.

        Args:
            x_feat: Stacked sampled rotations ``[2, E, d, d]``
                (``[0]`` = destination maps, ``[1]`` = source maps).
            edge_index: Graph connectivity ``[2, E]`` (unused; API parity).

        Returns:
            tuple: ``(identity, cross_map)`` -- ``self_map`` is the identity
            because :math:`F^\top F = I` for rotations.
        """
        maps_dst, maps_src = x_feat.unbind(0)
        cross_map = torch.matmul(maps_dst.transpose(-2, -1), maps_src)
        num_edges, d = cross_map.size(0), cross_map.size(-1)
        identity = torch.eye(d, device=cross_map.device, dtype=cross_map.dtype).expand(
            num_edges, d, d
        )
        return self._append_fixed_block(identity, cross_map)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        """Builds orthogonal Laplacian messages from normalised products.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Scalar degree inverse ``D^{-1}_dst`` broadcast [E, 1, 1].
            cross_map: Normalised ``D^{-1/2} F_dst^T F_src D^{-1/2}`` [E, d, d].

        Returns:
            torch.Tensor: Per-edge Laplacian messages [E, d, f].
        """
        return self_map * z_dst - torch.matmul(cross_map, z_src)


__all__ = [
    "BaseBSNNConv",
    "DiagonalBSNNConv",
    "GeneralBSNNConv",
    "LowRankBSNNConv",
    "OrthogonalBSNNConv",
]
