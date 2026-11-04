# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous Cooperative Sheaf Neural Network (Het-CSNN) model."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from hetsheaf_mpnn.csnn.csnn_layers import HetCSNNConv
from sheaf_mpnn.csnn import CSNNVariant


class HetCSNNModel(nn.Module):
    """Heterogeneous Cooperative Sheaf Neural Network for node prediction.

    Follows the same encoder -> layers -> decoder pipeline as ``CSNNModel``,
    but each cooperative layer's per-node conformal sheaf learners consume
    node-type one-hots (``use_node_types``) and, on edge-feature-aware
    backbones, edge-type one-hots (``use_edge_types``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: CSNNVariant = CSNNVariant.CONFORMAL,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        orth_strategy: str = "householder",
        clamp_val: float = 10.0,
        left_weights: bool = True,
        right_weights: bool = True,
        use_act: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = False,
        csnn_head_layers: int = 1,
        gnn_layers: int = 0,
        gnn_hidden: int = 16,
        gnn_default: int = 1,
        gnn_type: str = "SAGE",
        gnn_residual: bool = False,
        linear_emb: bool = False,
        pe_size: int = 0,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ):
        super().__init__()
        if stalk_dim <= 1:
            raise ValueError("stalk_dim must be greater than 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetCSNN layer")
        if csnn_head_layers <= 0:
            raise ValueError("csnn_head_layers must be positive")
        if pe_size < 0:
            raise ValueError("pe_size must be non-negative")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pe_size = pe_size
        self.use_act = use_act
        self.normalize_output = normalize_output
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        self.layers = nn.ModuleList(
            [
                HetCSNNConv(
                    stalk_dim=stalk_dim,
                    hidden_dim=hidden_dim,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    variant_conformal=variant.conformal,
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
                    pe_size=pe_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(context_dim) for _ in range(num_layers)])
            if layer_norm
            else None
        )
        self.batch_norms = (
            nn.ModuleList(
                [nn.BatchNorm1d(context_dim, affine=True) for _ in range(num_layers)]
            )
            if batch_norm
            else None
        )

        head: list[nn.Module] = []
        for _ in range(csnn_head_layers - 1):
            head.append(nn.Linear(context_dim, context_dim))
            head.append(nn.GELU())
        self.head: nn.Module = nn.Sequential(*head) if head else nn.Identity()
        self.decoder: nn.Module = nn.Linear(context_dim, out_channels)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetCSNNConv)
            layer.reset_parameters()
        for norms in (self.layer_norms, self.batch_norms):
            if norms is not None:
                for norm in norms:
                    assert isinstance(norm, nn.LayerNorm | nn.BatchNorm1d)
                    norm.reset_parameters()
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        # `decoder` may have been replaced with nn.Identity for graph-level
        # pooling, which has no parameters to reset.
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_types: Tensor | None = None,
        edge_types: Tensor | None = None,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
    ) -> Tensor:
        """Runs the het CSNN encoder, cooperative sheaf layers, and decoder.

        Args:
            x: Raw node features ``[N, in_channels]``.
            edge_index: Graph connectivity ``[2, E]``.
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.
            edge_attr: Optional edge features, forwarded to the sheaf learners
                alongside the edge-type one-hots.
            pe: Optional positional encoding ``[N, pe_size]``, concatenated to
                the sheaf-learner input before the node-type one-hots.

        Returns:
            Node outputs ``[N, out_channels]``, or the flat stalk
            representation when ``decoder`` has been replaced with
            ``nn.Identity``.
        """
        if pe is not None and pe.size(-1) != self.pe_size:
            raise ValueError(
                f"pe has width {pe.size(-1)} but pe_size is {self.pe_size}"
            )

        num_nodes = x.size(0)
        x = self.encoder(self.input_dropout_layer(x))
        if self.use_act:
            x = F.gelu(x)

        x_stalk = x.view(num_nodes, self.stalk_dim, self.hidden_dim)
        for i, layer in enumerate(self.layers):
            # Drop once: the sheaf learners and the layer's skip term must see
            # the same mask, as in the reference implementation.
            x_flat = self.dropout_layer(x_stalk.reshape(num_nodes, -1))
            x_stalk = x_flat.view(num_nodes, self.stalk_dim, self.hidden_dim)
            out = layer(
                x_flat, x_stalk, edge_index, node_types, edge_types, edge_attr, pe
            )
            if self.layer_norms is not None:
                out = self.layer_norms[i](out)
            if self.batch_norms is not None:
                out = self.batch_norms[i](out)
            x_stalk = out.view(num_nodes, self.stalk_dim, self.hidden_dim)

        x_out = x_stalk.reshape(num_nodes, -1)
        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)
        return self.decoder(self.head(x_out))


__all__ = ["HetCSNNModel"]
