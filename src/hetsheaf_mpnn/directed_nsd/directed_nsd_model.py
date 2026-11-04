# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.directed_nsd.directed_nsd_layers import (
    BaseHetDirectedNSDConv,
    DiagonalHetDirectedNSDConv,
    GeneralHetDirectedNSDConv,
    LowRankHetDirectedNSDConv,
    OrthogonalHetDirectedNSDConv,
)


class HetDirectedNSDVariant(Enum):
    """Variants for heterogeneous directed NSD models."""

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            HetDirectedNSDVariant.DIAGONAL: DiagonalHetDirectedNSDConv,
            HetDirectedNSDVariant.GENERAL: GeneralHetDirectedNSDConv,
            HetDirectedNSDVariant.ORTHOGONAL: OrthogonalHetDirectedNSDConv,
            HetDirectedNSDVariant.LOW_RANK: LowRankHetDirectedNSDConv,
            HetDirectedNSDVariant.GENERAL_ATTENTION: GeneralHetDirectedNSDConv,
            HetDirectedNSDVariant.ORTHOGONAL_ATTENTION: OrthogonalHetDirectedNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {HetDirectedNSDVariant.DIAGONAL, HetDirectedNSDVariant.LOW_RANK}:
            return {}
        if self in {
            HetDirectedNSDVariant.GENERAL_ATTENTION,
            HetDirectedNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth"] = "cayley",
        rank: int = 1,
    ) -> dict[str, Any]:
        """Build complete kwargs dict for layer instantiation."""
        kwargs = self.layer_kwargs.copy()
        if self in {
            HetDirectedNSDVariant.ORTHOGONAL,
            HetDirectedNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            kwargs["orth_strategy"] = orth_strategy
        if self == HetDirectedNSDVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


class HetDirectedNSDModel(nn.Module):
    """Heterogeneous Directed Neural Sheaf Diffusion (Het-Dir-NSD) model.

    Extends HetNSD to directed graphs by separately aggregating from src→dst
    and dst→src directions with independent restriction maps and sheaf degree
    normalizations per direction, following the Dir-GNN approach. Each layer's
    restriction-map MLP incorporates node/edge type information via the het
    context flags (use_features, use_node_types, use_edge_types, ensemble).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetDirectedNSDVariant = HetDirectedNSDVariant.ORTHOGONAL,
        alpha: float = 0.5,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_self_loops: bool = True,
        orth_strategy: Literal["cayley", "fasth"] = "cayley",
        rank: int = 1,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        learn_alpha: bool = False,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetDirectedNSD layer")

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

        variant_kwargs = variant.build_kwargs(orth_strategy=orth_strategy, rank=rank)

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_self_loops=add_self_loops,
                    learn_alpha=learn_alpha,
                    **variant_kwargs,
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
            assert isinstance(layer, BaseHetDirectedNSDConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Applies het directed NSD encoder, layers, and decoder.

        Args:
            x (Tensor): Raw node features [N, in_channels].
            edge_index (LongTensor): Directed graph connectivity [2, E].
            node_types (LongTensor, optional): Node-type indices [N].
            edge_types (LongTensor, optional): Edge-type indices [E].

        Returns:
            Tensor: Node outputs [N, out_channels].
        """
        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []
        for layer in self.layers:
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index, node_types, edge_types)
            if self.jknet:
                h = x_stalk.reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk.reshape(x_stalk.size(0), -1)

        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["HetDirectedNSDVariant", "HetDirectedNSDModel"]
