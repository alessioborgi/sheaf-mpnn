# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Shared sheaf-transport component for the set-based convolutions.

Steps 1-2 of the SheafDeepSet/SheafGIN/SheafSetTransformer pipeline -- generate
the restriction maps per directed edge and carry neighbour stalks into the
shared edge frame via the cross map F_u^T F_v -- are identical across all three.
Only what happens to the transported descriptors differs (mu-then-sum for
SheafDeepSet/SheafGIN, attention pooling for SheafSetTransformer).

``SheafTransport`` packages that shared half as a standalone ``nn.Module``. The
convolutions *hold* one (composition) rather than inherit it, so each conv keeps
single inheritance from its natural base (``MessagePassing`` for the sum-based
convs, plain ``nn.Module`` for the attention one) with no shared base class.
"""

from typing import Literal

import torch
from torch import nn

from sheaf_mpnn.utils import cayley, householder, init_map_generator_

MapVariant = Literal["diagonal", "general", "orthogonal", "low_rank"]


class SheafTransport(nn.Module):
    """Map generator + cross map + neighbour transport, shared by all set convs.

    Owns ``map_generator`` and exposes ``cross_map`` / ``transport_neighbors``;
    carries no aggregation policy (sum vs attention) and no PyG coupling.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        map_variant: MapVariant = "general",
        rank: int = 1,
        orth_strategy: Literal["cayley", "fasth"] = "cayley",
        clamp_val: float = 10.0,
    ):
        super().__init__()
        if map_variant not in {"diagonal", "general", "orthogonal", "low_rank"}:
            raise ValueError(
                "map_variant must be one of diagonal/general/orthogonal/low_rank,"
                f" got {map_variant!r}"
            )
        if map_variant == "orthogonal" and orth_strategy not in {"cayley", "fasth"}:
            raise ValueError(
                f"orth_strategy must be 'cayley' or 'fasth', got {orth_strategy!r}"
            )
        if map_variant == "low_rank" and rank <= 0:
            raise ValueError("rank must be positive")

        self.stalk_dim = stalk_dim
        self.in_channels = in_channels
        self.context_dim = (
            context_dim if context_dim is not None else stalk_dim * in_channels
        )
        self.map_variant = map_variant
        self.rank = rank
        self.orth_strategy = orth_strategy
        self.clamp_val = clamp_val

        # Number of raw outputs per node from the map generator MLP.
        if map_variant == "diagonal":
            map_out = stalk_dim
        elif map_variant == "orthogonal":
            # Cayley: d*(d-1)//2 skew-symmetric entries; fasth: d*d Householder params.
            map_out = (
                stalk_dim * stalk_dim
                if orth_strategy == "fasth"
                else (stalk_dim * (stalk_dim - 1)) // 2
            )
        elif map_variant == "low_rank":
            map_out = 2 * stalk_dim * rank
        else:  # general
            map_out = stalk_dim * stalk_dim

        self.map_generator = nn.Sequential(
            nn.Linear(2 * self.context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, map_out),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Init the map generator so the cross map F_u^T F_v starts near identity.

        Delegates to the shared ``init_map_generator_``; see its docstring for why a
        blanket near-zero init only suits the orthogonal variant.
        """
        init_map_generator_(self.map_generator, self.stalk_dim, self.map_variant)

    def _bidirectional_input(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch both edge orientations in one MLP forward pass, then split."""
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

    def cross_map(self, x_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute F_u^T F_v per edge.

        Returns:
            [E, d]    for diagonal variant (diagonal elements of the cross map).
            [E, d, d] for general / orthogonal / low_rank variants.
        """
        d = self.stalk_dim
        phi_dst, phi_src = self._bidirectional_input(x_feat, edge_index)

        if self.map_variant == "diagonal":
            # F = diag(w), so F_dst^T F_src = diag(w_dst * w_src)
            return phi_dst * phi_src  # [E, d]

        if self.map_variant == "orthogonal":
            if self.orth_strategy == "fasth":
                w_dst = householder(phi_dst, d)
                w_src = householder(phi_src, d)
            else:
                w_dst = cayley(phi_dst, d, self.clamp_val)
                w_src = cayley(phi_src, d, self.clamp_val)
            return torch.matmul(w_dst.transpose(-2, -1), w_src)  # [E, d, d]

        if self.map_variant == "low_rank":
            # F = left @ right^T, so F^T = right @ left^T
            left_dst, right_dst = phi_dst.view(-1, d, 2 * self.rank).chunk(2, dim=-1)
            left_src, right_src = phi_src.view(-1, d, 2 * self.rank).chunk(2, dim=-1)
            gram = torch.matmul(left_dst.transpose(-2, -1), left_src)  # [E, r, r]
            return torch.matmul(
                right_dst, torch.matmul(gram, right_src.transpose(-2, -1))
            )  # [E, d, d]

        # general: unconstrained phi -> d*d matrix
        phi_dst = phi_dst.view(-1, d, d)
        phi_src = phi_src.view(-1, d, d)
        return torch.matmul(phi_dst.transpose(-2, -1), phi_src)  # [E, d, d]

    def transport_neighbors(
        self, cross_map: torch.Tensor, x_stalk_src: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross map F_u^T F_v to transported neighbour stalk x_stalk_src.

        Args:
            cross_map:     [E, d] for diagonal, [E, d, d] otherwise.
            x_stalk_src:   [E, d, f] neighbour stalk features.

        Returns:
            Transported features [E, d, f].
        """
        if self.map_variant == "diagonal":
            # diagonal maps scale each d-channel independently
            return cross_map[:, :, None] * x_stalk_src  # [E, d, f]
        return torch.matmul(cross_map, x_stalk_src)  # [E, d, f]


__all__ = ["SheafTransport", "MapVariant"]
