# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import add_self_loops, softmax

from hetsheaf_mpnn.base_conv import BaseHetSheafConv
from sheaf_mpnn.utils import cayley, householder


class HetSheafAttnConv(BaseHetSheafConv):
    """Heterogeneous Sheaf Attention Network convolution layer.

    Extends SheafAttnConv with het context: node/edge type information is
    incorporated into **both** the attention score computation and the
    restriction-map generators, so that type heterogeneity shapes both which
    neighbors are attended to and what restriction maps are learned.

    Het context flags:

    - ``use_features``: include node features in per-node attention projection
      and in the per-edge map-generator MLP context.
    - ``use_node_types``: include node-type one-hots in per-node attention
      projection and in the per-edge map-generator MLP context.
    - ``use_edge_types``: include edge-type one-hots in the per-edge map-generator
      MLP context (edge-level; not applicable to attention scores).
    - ``ensemble``: use independent map-generators per edge type (per head). In
      ensemble mode, edge types are not included in the MLP context as they are
      discriminated by the MLP index.

    Attention projection input dim = attn_ctx_dim. Map-generator input dim (per
    head)::

        2 * attn_ctx_dim + (num_edge_types if (use_edge_types and not ensemble) else 0)
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 1,
        leaky_relu_slope: float = 0.2,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        clamp_val: float = 10.0,
        residual: bool = False,
        orth_strategy: str = "cayley",
        add_eps: bool = False,
    ):
        super().__init__(
            stalk_dim,
            [in_channels],
            hidden_dim,
            context_dim=context_dim,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual
        self.orth_strategy = orth_strategy
        self.clamp_val = clamp_val
        self.add_eps = add_eps
        if add_eps and residual:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

        # Use ELU activation (matching SheafAttnConv).
        self.sigma = nn.ELU()

        # Attention projection: per-node, includes node type if requested.
        self._attn_ctx_dim = (self.context_dim if use_features else 0) + (
            num_node_types if use_node_types else 0
        )
        self.attn_projection = nn.Linear(self._attn_ctx_dim, hidden_dim, bias=False)
        self.attn_score_vec = nn.Parameter(torch.empty(num_heads, 2 * hidden_dim))

        # Map-generator MLP input: two endpoints (het-augmented) + edge type.
        # Each generator receives [ctx_i, ctx_j, edge_type_oh] for each head.
        self._map_gen_in = 2 * self._attn_ctx_dim + (
            num_edge_types if use_edge_types else 0
        )

        num_map_params = (
            stalk_dim * stalk_dim
            if orth_strategy == "fasth"
            else (stalk_dim * (stalk_dim - 1)) // 2
        )

        if ensemble:
            # One nn.ModuleList of num_heads per edge type.
            # Stored as a flat ModuleList of shape [num_edge_types * num_heads],
            # indexed as [edge_type * num_heads + head].
            # Input dim is 2 * _attn_ctx_dim (not bare context_dim) so that
            # node-type one-hots are included when use_node_types=True, matching
            # the node_het_ctx_i/j tensors passed by message().
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self._attn_ctx_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, num_map_params),
                    )
                    for _ in range(num_edge_types * num_heads)
                ]
            )
            self._ensemble_stride = num_heads
        else:
            self.map_generators = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self._map_gen_in, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, num_map_params),
                    )
                    for _ in range(num_heads)
                ]
            )

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.xavier_uniform_(self.attn_score_vec)
        nn.init.xavier_uniform_(self.attn_projection.weight)

    def _to_ortho(self, map_params: torch.Tensor) -> torch.Tensor:
        if self.orth_strategy == "fasth":
            return householder(map_params, self.stalk_dim)
        return cayley(map_params, self.stalk_dim, self.clamp_val)

    def _build_node_het_ctx(
        self, x_feat: torch.Tensor, node_types: torch.Tensor | None
    ) -> torch.Tensor:
        """Returns [N, attn_ctx_dim] het-augmented per-node context."""
        parts = []
        if self.use_features:
            parts.append(x_feat)

        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
            else:
                nt_oh = torch.zeros(
                    x_feat.size(0),
                    self.num_node_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts.append(nt_oh)

        if not parts:  # pragma: no cover
            return torch.empty(
                x_feat.size(0), 0, device=x_feat.device, dtype=x_feat.dtype
            )

        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one sheaf attention layer.

        Args:
            x_feat: [N, context_dim]
            x_stalk: [N, stalk_dim, in_channels]
            edge_index: [2, E]
            node_types: [N] or None
            edge_types: [E] or None

        Returns:
            Updated stalk [N, stalk_dim, in_channels].
        """
        N = x_stalk.size(0)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=N)
        E_total = edge_index_sl.size(1)

        z = self._apply_stalk_transform(x_stalk)

        # Build het-augmented per-node context for attention.
        node_het_ctx = self._build_node_het_ctx(x_feat, node_types)  # [N, attn_ctx_dim]

        node_proj = self.attn_projection(node_het_ctx)  # [N, hidden_dim]
        attn_dst = node_proj @ self.attn_score_vec[:, : self.hidden_dim].T  # [N, H]
        attn_src = node_proj @ self.attn_score_vec[:, self.hidden_dim :].T  # [N, H]

        # Build edge-type one-hots for map generators (with self-loops zeroed).
        if self.use_edge_types:
            if edge_types is not None:
                real_et_oh = F.one_hot(edge_types, self.num_edge_types).to(x_feat.dtype)
                self_loop_et = torch.zeros(
                    N, self.num_edge_types, device=x_feat.device, dtype=x_feat.dtype
                )
                edge_het_ctx = torch.cat(
                    [real_et_oh, self_loop_et], dim=0
                )  # [E_total, E_t]
            else:
                edge_het_ctx = torch.zeros(
                    E_total,
                    self.num_edge_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
        else:
            edge_het_ctx = None

        z_src = z[edge_index_sl[0]]

        propagated = self.propagate(  # ty: ignore[missing-argument]
            edge_index_sl,  # ty: ignore[invalid-argument-type]
            z_src=z_src,
            node_het_ctx=node_het_ctx,
            attn_dst=attn_dst,
            attn_src=attn_src,
            edge_het_ctx=edge_het_ctx,
            edge_types_idx=edge_types,
            size=(N, N),
        )

        if self.residual:
            skip = (
                (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
                if self.add_eps
                else x_stalk
            )
            return skip + self.sigma(propagated - z)
        return self.sigma(propagated)

    def message(  # ty: ignore[invalid-method-override]
        self,
        z_src,
        node_het_ctx_i,
        node_het_ctx_j,
        attn_dst_i,
        attn_src_j,
        edge_het_ctx,
        edge_types_idx,
        index,
        size_i,
    ):
        """Attention-weighted orthogonal transport message for each edge.

        node_het_ctx_i/j are auto-expanded per-node het contexts.
        edge_het_ctx is edge-level (not auto-indexed).
        """
        attn_weights = softmax(
            self.leaky_relu(attn_dst_i + attn_src_j), index, num_nodes=size_i
        )  # [E_total, num_heads]

        if self.ensemble:
            # Per-edge-type dispatch: use edge_types_idx to select generator.
            # edge_types_idx is a node-level tensor passed as-is (no _i/_j suffix).
            transport_maps = self._ensemble_transport(
                node_het_ctx_i, node_het_ctx_j, edge_types_idx
            )
        else:
            # Build map-generator input for each edge: [ctx_i, ctx_j, edge_het]
            if edge_het_ctx is not None:
                restr_input_i = torch.cat(
                    [node_het_ctx_i, node_het_ctx_j, edge_het_ctx], dim=-1
                )
                restr_input_j = torch.cat(
                    [node_het_ctx_j, node_het_ctx_i, edge_het_ctx], dim=-1
                )
            else:
                restr_input_i = torch.cat([node_het_ctx_i, node_het_ctx_j], dim=-1)
                restr_input_j = torch.cat([node_het_ctx_j, node_het_ctx_i], dim=-1)

            transport_maps = torch.stack(
                [
                    torch.matmul(
                        self._to_ortho(gen(restr_input_i)).transpose(-2, -1),
                        self._to_ortho(gen(restr_input_j)),
                    )
                    for gen in self.map_generators
                ],
                dim=1,
            )  # [E_total, num_heads, d, d]

        weighted = (attn_weights.unsqueeze(-1).unsqueeze(-1) * transport_maps).mean(
            dim=1
        )  # [E_total, d, d]

        return torch.matmul(weighted, z_src)

    def _ensemble_transport(
        self,
        ctx_i: torch.Tensor,
        ctx_j: torch.Tensor,
        edge_types_idx,
    ) -> torch.Tensor:
        """Computes transport maps for ensemble mode (per-edge-type MLPs per head)."""
        E = ctx_i.size(0)
        H = self.num_heads
        d = self.stalk_dim

        restr_input_i = torch.cat([ctx_i, ctx_j], dim=-1)
        restr_input_j = torch.cat([ctx_j, ctx_i], dim=-1)

        transport_maps = torch.zeros(E, H, d, d, device=ctx_i.device, dtype=ctx_i.dtype)

        if edge_types_idx is None:
            for h in range(H):
                gen = self.map_generators[h]
                transport_maps[:, h] = torch.matmul(
                    self._to_ortho(gen(restr_input_i)).transpose(-2, -1),
                    self._to_ortho(gen(restr_input_j)),
                )
            return transport_maps

        # edge_types_idx may be shorter than E (no self-loops);
        # pad with 0 for self-loops.
        E_real = edge_types_idx.size(0)
        et_full = torch.zeros(E, dtype=torch.long, device=ctx_i.device)
        et_full[:E_real] = edge_types_idx

        unique_types, counts = torch.unique(et_full, return_counts=True)
        sorted_idx = torch.argsort(et_full)
        splits = sorted_idx.split(counts.tolist())

        for split, type_val in zip(splits, unique_types.tolist(), strict=False):
            for h in range(H):
                gen = self.map_generators[type_val * H + h]
                W_i = self._to_ortho(gen(restr_input_i[split]))
                W_j = self._to_ortho(gen(restr_input_j[split]))
                transport_maps[split, h] = torch.matmul(W_i.transpose(-2, -1), W_j)

        return transport_maps


__all__ = ["HetSheafAttnConv"]
