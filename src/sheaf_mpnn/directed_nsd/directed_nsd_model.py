# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.directed_nsd.directed_nsd_layers import (
    BaseDirectedNSDConv,
    DiagonalDirectedNSDConv,
    GeneralDirectedNSDConv,
    LowRankDirectedNSDConv,
    OrthogonalDirectedNSDConv,
)


class DirectedNSDVariant(Enum):
    """Variants for directed NSD models."""

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            DirectedNSDVariant.DIAGONAL: DiagonalDirectedNSDConv,
            DirectedNSDVariant.GENERAL: GeneralDirectedNSDConv,
            DirectedNSDVariant.ORTHOGONAL: OrthogonalDirectedNSDConv,
            DirectedNSDVariant.LOW_RANK: LowRankDirectedNSDConv,
            DirectedNSDVariant.GENERAL_ATTENTION: GeneralDirectedNSDConv,
            DirectedNSDVariant.ORTHOGONAL_ATTENTION: OrthogonalDirectedNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {DirectedNSDVariant.DIAGONAL, DirectedNSDVariant.LOW_RANK}:
            return {}
        if self in {
            DirectedNSDVariant.GENERAL_ATTENTION,
            DirectedNSDVariant.ORTHOGONAL_ATTENTION,
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
            DirectedNSDVariant.ORTHOGONAL,
            DirectedNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            kwargs["orth_strategy"] = orth_strategy
        if self == DirectedNSDVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


class DirectedNSDModel(nn.Module):
    """Directed Neural Sheaf Diffusion (Dir-NSD) model.

    Extends NSD to directed graphs by separately aggregating from src→dst and dst→src
    directions with independent restriction maps and sheaf degree normalizations per
    direction, following the Dir-GNN approach.

    The wrapper lifts raw node features into stalk features, applies a stack of directed
    NSD convolution layers with separate src/dst transformations, and decodes the
    flattened stalk representation back to the requested output dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: DirectedNSDVariant = DirectedNSDVariant.ORTHOGONAL,
        alpha: float = 0.5,
        add_self_loops: bool = True,
        orth_strategy: Literal["cayley", "fasth"] = "cayley",
        rank: int = 1,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        learn_alpha: bool = False,
    ):
        """Initializes a directed NSD model for node-level prediction.

        Args:
            in_channels: Raw input features per node.
            out_channels: Output channels per node (e.g. num classes).
            stalk_dim: Stalk dimension [d]. Node shape: [stalk_dim, hidden_dim].
            hidden_dim: Feature dimension per stalk channel. State size: d*f.
            num_layers: Number of directed NSD layers (must be positive).
            variant: Restriction-map family (DIAGONAL, GENERAL, ORTHOGONAL,
                LOW_RANK).
            alpha: Balance parameter for direction blending (0=dst→src,
                1=src→dst). Defaults to 0.5.
            add_self_loops: If True, augment degree normalization.
            orth_strategy: "cayley" or "fasth" for orthogonal maps.
            rank: Rank for low-rank restriction maps. Defaults to 1.
            input_dropout: Dropout for raw input features. Defaults to 0.0.
            dropout: Dropout for stalk features between layers. Defaults to 0.0.
            normalize_output: If True, L2-normalise before decoder.
            jknet: If True, concatenate all layer outputs (jumping knowledge).
            learn_alpha: If True, alpha becomes a learnable parameter.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one DirectedNSD layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        # Build variant-specific kwargs
        variant_kwargs = variant.build_kwargs(orth_strategy=orth_strategy, rank=rank)

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
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
            assert isinstance(layer, BaseDirectedNSDConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Applies directed NSD encoder, layers, and decoder.

        Args:
            x: Raw node features [num_nodes, in_channels].
            edge_index: Directed graph connectivity [2, num_edges].

        Returns:
            Node outputs [num_nodes, out_channels].
        """
        if x.dim() != 2 or x.size(1) != self.encoder.in_features:
            raise ValueError(
                f"x must have shape [num_nodes, {self.encoder.in_features}],"
                f" got {tuple(x.shape)}"
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges],"
                f" got {tuple(edge_index.shape)}"
            )

        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []
        for layer in self.layers:
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index)
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


__all__ = ["DirectedNSDVariant", "DirectedNSDModel"]
