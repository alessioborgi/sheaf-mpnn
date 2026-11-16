# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Encoder/convolution/decoder models for undirected hypergraph sheaves."""

from enum import Enum, auto
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.hypergraph_sheaf_conv import BaseHypergraphSheafConv
from sheaf_mpnn.nsd.hsnn_layers import (
    DiagonalHSNNConv,
    GeneralHSNNConv,
    LowRankHSNNConv,
    OrthogonalHSNNConv,
)


class HSNNVariant(Enum):
    """Restriction-map families for HSNN; independent of graph NSD variants."""

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self) -> type[BaseHypergraphSheafConv]:
        return {
            HSNNVariant.DIAGONAL: DiagonalHSNNConv,
            HSNNVariant.GENERAL: GeneralHSNNConv,
            HSNNVariant.ORTHOGONAL: OrthogonalHSNNConv,
            HSNNVariant.LOW_RANK: LowRankHSNNConv,
        }[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        return {}

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
    ) -> dict[str, Any]:
        """Select only the arguments relevant to this restriction-map family."""
        kwargs = self.layer_kwargs
        if self == HSNNVariant.ORTHOGONAL:
            kwargs["orth_strategy"] = orth_strategy
        if self == HSNNVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


class HSNNModel(nn.Module):
    """Lift node features, propagate over hyperedges, then decode node logits.

    Args:
        in_channels: Positive number of raw input features.
        out_channels: Positive output width, e.g. the number of node classes.
        stalk_dim: Positive stalk dimension d.
        hidden_dim: Positive feature width f inside each stalk entry.
        num_layers: Positive number of HSNN convolutions.
        variant: Restriction-map family; defaults to diagonal maps.
        input_dropout: Dropout probability before the encoder.
        dropout: Dropout probability after encoder ELU and on layer inputs.
        residual: Add each layer's input before its output ELU.
        degree_regularization: Nonnegative tau, zero for pseudoinverse degrees.
        orth_strategy: Strategy passed only to the orthogonal variant.
        rank: Factor width passed only to the low-rank variant.

    Notes:
        Uses real, unit-weight, undirected incidences for a single full-batch
        hypergraph. Every layer predicts fresh maps from its current states.
        The ordinary graph experiment CLI is not used for incidence data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HSNNVariant = HSNNVariant.DIAGONAL,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, stalk_dim, hidden_dim, num_layers) <= 0:
            raise ValueError(
                "channel widths, stalk_dim and num_layers must be positive"
            )
        if not isinstance(variant, HSNNVariant):
            raise ValueError("variant must be an HSNNVariant")
        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        context_dim = stalk_dim * hidden_dim
        self.encoder = nn.Linear(in_channels, context_dim)
        self.layers = nn.ModuleList(
            [
                variant.layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    dropout=dropout,
                    residual=residual,
                    degree_regularization=degree_regularization,
                    **variant.build_kwargs(orth_strategy=orth_strategy, rank=rank),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(context_dim, out_channels)

    def reset_parameters(self) -> None:
        """Reset encoder, decoder, transforms and every restriction-map learner."""
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseHypergraphSheafConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
        task: str = "node",
    ) -> torch.Tensor:
        """Compute raw node logits for a single undirected hypergraph.

        Args:
            x: Real features [N > 0, in_channels], matching model device/dtype.
            hyperedge_index: Long indices [2, K], node IDs followed by hyperedge
                IDs in separate numbering spaces. Duplicate incidences are invalid.
            task: Prediction domain, node or hyperedge; hyperedges use sum readout.
            num_hyperedges: Optional M, including trailing empty hyperedges. If
                omitted, inferred from incidences (zero for an empty index).

        Returns:
            Node logits [N, out_channels], without a softmax.
        """
        if task not in ("node", "hyperedge"):
            raise ValueError("task must be node or hyperedge")
        if x.ndim != 2 or x.size(0) == 0 or x.size(1) != self.encoder.in_features:
            raise ValueError("x must have shape [N > 0, in_channels]")
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("x must have dtype float32 or float64")
        if (
            x.dtype != self.encoder.weight.dtype
            or x.device != self.encoder.weight.device
        ):
            raise ValueError("x must match the model's device and dtype")
        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        x_stalk = x_stalk.reshape(x.size(0), self.stalk_dim, self.hidden_dim)
        for layer in self.layers:
            x_feat = x_stalk.reshape(x.size(0), -1)
            x_stalk = layer(
                x_feat, x_stalk, hyperedge_index, num_hyperedges=num_hyperedges
            )
        embedding = x_stalk.reshape(x.size(0), -1)
        if task == "hyperedge":
            from torch_geometric.utils import scatter

            embedding = scatter(
                embedding[hyperedge_index[0]],
                hyperedge_index[1],
                dim=0,
                dim_size=num_hyperedges,
                reduce="sum",
            )
        return self.decoder(embedding)


__all__ = ["HSNNVariant", "HSNNModel"]
