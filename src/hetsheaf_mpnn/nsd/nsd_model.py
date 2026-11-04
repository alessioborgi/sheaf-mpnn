# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.nsd.nsd_layers import (
    BaseHetNSDConv,
    DiagonalHetNSDConv,
    GeneralHetNSDConv,
    LowRankHetNSDConv,
    OrthogonalHetNSDConv,
)


class HetNSDVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            HetNSDVariant.DIAGONAL: DiagonalHetNSDConv,
            HetNSDVariant.GENERAL: GeneralHetNSDConv,
            HetNSDVariant.ORTHOGONAL: OrthogonalHetNSDConv,
            HetNSDVariant.GENERAL_ATTENTION: GeneralHetNSDConv,
            HetNSDVariant.ORTHOGONAL_ATTENTION: OrthogonalHetNSDConv,
            HetNSDVariant.LOW_RANK: LowRankHetNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {HetNSDVariant.DIAGONAL, HetNSDVariant.LOW_RANK}:
            return {}
        if self in {
            HetNSDVariant.GENERAL_ATTENTION,
            HetNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}


class HetNSDModel(nn.Module):
    """Heterogeneous Neural Sheaf Diffusion (HetNSD) model.

    Identical encoder-layers-decoder pipeline to NSDModel, but each diffusion
    layer incorporates node/edge type information into the restriction-map MLP
    via the het context flags (use_features, use_node_types, use_edge_types,
    ensemble).

    The encoder is shared across all node types (shared cochain projection),
    matching the existing heterogeneous experiment pipeline which flattens
    HeteroData via PyG's to_homogeneous() before passing it here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetNSDVariant = HetNSDVariant.GENERAL,
        alpha: float = 1.0,
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
            raise ValueError("must have at least one HetNSD layer")

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
        if variant == HetNSDVariant.ORTHOGONAL:
            extra_kwargs["orth_strategy"] = orth_strategy
        if variant == HetNSDVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

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
            assert isinstance(layer, BaseHetNSDConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het-NSD diffusion layers, and decoder.

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


__all__ = ["HetNSDVariant", "HetNSDModel"]
