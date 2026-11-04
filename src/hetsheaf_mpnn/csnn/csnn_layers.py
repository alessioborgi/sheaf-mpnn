# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous Cooperative Sheaf Neural Network convolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from sheaf_mpnn.csnn.csnn_layers import CSNNConv


class HetCSNNConv(CSNNConv):
    """Heterogeneous cooperative sheaf convolution.

    Injects node/edge-type context into the per-node conformal sheaf learners
    of ``CSNNConv``:

    * ``use_node_types`` -- one-hot node types are appended to the learner
      input via the positional-encoding channel the learner already supports
      (the het widening is added on top of any user ``pe_size``).
    * ``use_edge_types`` -- one-hot edge types are forwarded as edge features
      to the learner backbone. Because only custom backbones consume
      ``edge_attr``, this flag requires ``gnn_layers > 0``, ``gnn_default=0``
      and a ``gnn_type`` whose convolution accepts edge features (e.g. GAT or
      NNConv); anything else would silently drop the edge types, so it is
      rejected eagerly.

    The CSNN learners are per-node GNNs on the full stalk features, so there
    is no ``use_features`` switch (features are the learner input itself) and
    no per-edge-type ``ensemble`` mode (the maps are not generated per edge).
    """

    def __init__(
        self,
        stalk_dim: int,
        hidden_dim: int,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        variant_conformal: bool = True,
        orth_strategy: str = "householder",
        clamp_val: float = 10.0,
        left_weights: bool = True,
        right_weights: bool = True,
        use_act: bool = True,
        dropout: float = 0.0,
        gnn_layers: int = 0,
        gnn_hidden: int = 16,
        gnn_default: int = 1,
        gnn_type: str = "SAGE",
        gnn_residual: bool = False,
        linear_emb: bool = False,
        pe_size: int = 0,
        bias: bool = False,
    ):
        if use_node_types and num_node_types <= 0:
            raise ValueError(
                f"num_node_types must be > 0 when use_node_types=True, "
                f"got num_node_types={num_node_types}"
            )
        if use_edge_types and num_edge_types <= 0:
            raise ValueError(
                f"num_edge_types must be > 0 when use_edge_types=True, "
                f"got num_edge_types={num_edge_types}"
            )

        het_pe = num_node_types if use_node_types else 0
        super().__init__(
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            variant_conformal=variant_conformal,
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
            left_weights=left_weights,
            right_weights=right_weights,
            use_act=use_act,
            dropout=dropout,
            gnn_layers=gnn_layers,
            gnn_hidden=gnn_hidden,
            gnn_default=gnn_default,
            gnn_type=gnn_type,
            gnn_residual=gnn_residual,
            linear_emb=linear_emb,
            pe_size=pe_size + het_pe,
            bias=bias,
        )
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_node_types = use_node_types
        self.use_edge_types = use_edge_types
        # Width of the caller-supplied positional encoding (the het one-hot
        # widening lives on top of it).
        self.base_pe_size = pe_size

        if use_edge_types and not (
            gnn_layers > 0
            and gnn_default == 0
            and self.out_maps_learner._accepts_edge_attr()
        ):
            raise ValueError(
                "use_edge_types=True requires a sheaf-learner backbone that "
                "consumes edge features: set gnn_layers > 0, gnn_default=0 and "
                "a gnn_type whose convolution accepts edge_attr (e.g. 'GAT' or "
                f"'NNConv'); got gnn_layers={gnn_layers}, "
                f"gnn_default={gnn_default}, gnn_type={gnn_type!r}."
            )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x_feat: Tensor,
        x_stalk: Tensor,
        edge_index: Tensor,
        node_types: Tensor | None = None,
        edge_types: Tensor | None = None,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
    ) -> Tensor:
        """Runs one het cooperative sheaf diffusion step.

        Args:
            x_feat: Flattened stalk features ``[N, d * hidden_dim]``.
            x_stalk: Stalk features ``[N, d, hidden_dim]``.
            edge_index: Graph connectivity ``[2, E]``.
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.
            edge_attr: Optional edge features, concatenated before the edge-type
                one-hots and forwarded to the sheaf learners.
            pe: Optional positional encoding ``[N, base_pe_size]``, concatenated
                before the node-type one-hots.

        Returns:
            Updated flattened stalk features ``[N, d * hidden_dim]``.
        """
        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
            else:
                nt_oh = x_feat.new_zeros(x_feat.size(0), self.num_node_types)
            pe = nt_oh if pe is None else torch.cat([pe, nt_oh], dim=-1)

        if self.use_edge_types:
            if edge_types is not None:
                et_oh = F.one_hot(edge_types, self.num_edge_types).to(x_feat.dtype)
            else:
                et_oh = x_feat.new_zeros(edge_index.size(1), self.num_edge_types)
            edge_attr = (
                et_oh if edge_attr is None else torch.cat([edge_attr, et_oh], dim=-1)
            )

        return super().forward(x_feat, x_stalk, edge_index, edge_attr, pe)


__all__ = ["HetCSNNConv"]
