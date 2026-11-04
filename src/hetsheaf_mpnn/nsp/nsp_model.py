# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.nsp.nsp_layers import (
    BaseHetNSPConv,
    DiagonalHetNSPConv,
    GeneralHetNSPConv,
    LowRankHetNSPConv,
    OrthogonalHetNSPConv,
)


class HetNSPVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            HetNSPVariant.DIAGONAL: DiagonalHetNSPConv,
            HetNSPVariant.GENERAL: GeneralHetNSPConv,
            HetNSPVariant.ORTHOGONAL: OrthogonalHetNSPConv,
            HetNSPVariant.GENERAL_ATTENTION: GeneralHetNSPConv,
            HetNSPVariant.ORTHOGONAL_ATTENTION: OrthogonalHetNSPConv,
            HetNSPVariant.LOW_RANK: LowRankHetNSPConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {HetNSPVariant.DIAGONAL, HetNSPVariant.LOW_RANK}:
            return {}
        if self in {
            HetNSPVariant.GENERAL_ATTENTION,
            HetNSPVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}


class HetNSPModel(nn.Module):
    """Heterogeneous Neural Sheaf Propagation (HetNSP) model.

    Second-order het-sheaf model using state augmentation [x_curr; x_hist].
    Het context flags control type-aware restriction-map generation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetNSPVariant = HetNSPVariant.GENERAL,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_self_loops: bool = False,
        orth_strategy: str = "cayley",
        rank: int = 1,
        add_eps: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetNSP layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        extra_kwargs = variant.layer_kwargs.copy()
        if variant == HetNSPVariant.ORTHOGONAL:
            extra_kwargs["orth_strategy"] = orth_strategy
        if variant == HetNSPVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseHetNSPConv), (
                f"Expected BaseHetNSPConv, got {type(layer).__name__}"
            )
            layer.reset_parameters()
        if hasattr(self.decoder, "reset_parameters"):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het-NSP propagation layers, and decoder.

        Args:
            x (Tensor): Raw node features [N, in_channels].
            edge_index (LongTensor): Graph connectivity [2, E].
            node_types (LongTensor, optional): Node-type indices [N].
            edge_types (LongTensor, optional): Edge-type indices [E].

        Returns:
            Tensor: Node outputs [N, out_channels].
        """
        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )
        # Augment with zero history.
        x_stalk = torch.cat([x_stalk, torch.zeros_like(x_stalk)], dim=1)  # [N, 2d, f]

        layer_reps = []
        for layer in self.layers:
            x_curr = x_stalk[:, : self.stalk_dim, :]
            x_feat = self.dropout_layer(x_curr.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index, node_types, edge_types)
            if self.jknet:
                h = x_stalk[:, : self.stalk_dim, :].reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk[:, : self.stalk_dim, :].reshape(x_stalk.size(0), -1)

        if self.normalize_output and not self.jknet:
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["HetNSPVariant", "HetNSPModel"]
