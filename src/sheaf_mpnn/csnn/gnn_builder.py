# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""GNN backbones for the CSNN conformal sheaf learner.

The CSNN restriction-map learner runs node features through a configurable GNN
(or a plain MLP) before projecting to orthogonal-map parameters.  This module
holds the layer-type lookup and stack builder for that backbone, kept separate
from ``csnn_layers`` so the sheaf math stays readable.

Unlike the rest of ``sheaf_mpnn`` -- where the restriction-map generator is a
single ``nn.Linear`` -- CSNN's generator is a multi-layer message-passing
network, so it needs a real builder.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GPSConv,
    MessagePassing,
    NNConv,
    SAGEConv,
    SGConv,
)


class SumGNN(MessagePassing):
    """Sum-aggregation convolution with separate self and neighbour weights.

    Computes ``gelu(W_s x_i + W_n sum_j x_j)``.  Kept from the reference CSNN
    implementation as a selectable ``gnn_type``.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.W_s = nn.Linear(in_channels, out_channels, bias=False)
        self.W_n = nn.Linear(in_channels, out_channels, bias=False)

    def reset_parameters(self) -> None:
        self.W_s.reset_parameters()
        self.W_n.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Aggregates neighbours and mixes them with the self representation.

        Args:
            x: Node features ``[num_nodes, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``.

        Returns:
            Node features ``[num_nodes, out_channels]``.
        """
        out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            x=x,
        )
        return F.gelu(self.W_s(x) + self.W_n(out))

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


_LAYER_TYPES: dict[str, type[nn.Module]] = {
    "GCN": GCNConv,
    "GAT": GATConv,
    "SAGE": SAGEConv,
    "SGC": SGConv,
    "GPS": GPSConv,
    "NNConv": NNConv,
    "SumGNN": SumGNN,
}

# Types whose constructor is the plain ``(in_channels, out_channels)`` form.
_SIMPLE_TYPES = frozenset({"GCN", "GAT", "SAGE", "SumGNN"})

GNN_TYPES = tuple(_LAYER_TYPES)


def get_layer_type(layer_type: str) -> type[nn.Module]:
    """Resolves a ``gnn_type`` string to its convolution class.

    Args:
        layer_type: One of ``GNN_TYPES``.

    Returns:
        The corresponding ``nn.Module`` subclass.

    Raises:
        ValueError: If ``layer_type`` is not a supported GNN type.
    """
    try:
        return _LAYER_TYPES[layer_type]
    except KeyError:
        raise ValueError(
            f"Unsupported GNN layer type: {layer_type!r}. Valid choices: "
            f"{list(GNN_TYPES)}"
        ) from None


def build_gnn(
    gnn_type: str,
    in_channels: int,
    out_channels: int,
    num_layers: int,
    hidden_channels: int | None = None,
    aggr: str = "add",
) -> nn.Module:
    """Builds the learner's GNN stack.

    When ``hidden_channels`` is ``None`` or ``num_layers == 1`` every layer maps
    ``in_channels -> out_channels``; otherwise the stack widens through
    ``hidden_channels`` and narrows again on the last layer.

    Args:
        gnn_type: One of ``GNN_TYPES``.
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        num_layers: Number of convolution layers. Must be positive.
        hidden_channels: Width of the intermediate layers, or ``None`` to keep
            every layer at ``in_channels -> out_channels``.
        aggr: Aggregation for ``NNConv``. Ignored by the other types.

    Returns:
        An ``nn.ModuleList`` of convolutions, except for ``SGC`` in the
        no-hidden case where a single ``SGConv`` with ``K=num_layers`` is
        returned (``SGConv`` folds its propagation depth into one layer).

    Raises:
        ValueError: If ``gnn_type`` is unsupported, ``num_layers < 1``, or
            ``GPS`` is requested with a width change (``GPSConv`` preserves its
            input width, so it only supports ``in_channels == out_channels``
            stacks -- the ``linear_emb`` learner path; the reference raises
            ``NotImplementedError`` for GPS here).
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    gnn = get_layer_type(gnn_type)
    layers = nn.ModuleList()

    if hidden_channels is None or num_layers == 1:
        if gnn_type == "GPS":
            if in_channels != out_channels:
                raise ValueError(
                    "GPSConv preserves its input width, so the GPS stack needs "
                    f"in_channels == out_channels (got {in_channels} -> "
                    f"{out_channels}); use linear_emb=True."
                )
            for _ in range(num_layers):
                layers.append(
                    GPSConv(
                        in_channels,
                        conv=GCNConv(in_channels, in_channels),
                        heads=2,
                        dropout=0.2,
                    )
                )
        elif gnn_type == "NNConv":
            # One shared edge network across layers, matching the reference.
            edge_net = nn.LazyLinear(in_channels * out_channels)
            for _ in range(num_layers):
                layers.append(NNConv(in_channels, out_channels, nn=edge_net, aggr=aggr))
        elif gnn_type == "SGC":
            return SGConv(in_channels, out_channels, K=num_layers)
        elif gnn_type in _SIMPLE_TYPES:
            for _ in range(num_layers):
                layers.append(gnn(in_channels, out_channels))
        return layers

    if gnn_type == "GPS":
        # A widening GPS stack would need plain Linear stages that the layer
        # runner cannot call with edge_index; the reference rejects GPS here too.
        raise ValueError(
            "GPS does not support a hidden_channels stack; use linear_emb=True, "
            "which builds a width-preserving GPS stack instead."
        )
    if gnn_type == "NNConv":
        layers.append(
            NNConv(
                in_channels,
                hidden_channels,
                nn=nn.LazyLinear(in_channels * hidden_channels),
                aggr=aggr,
            )
        )
        for _ in range(num_layers - 2):
            layers.append(
                NNConv(
                    hidden_channels,
                    hidden_channels,
                    nn=nn.LazyLinear(hidden_channels**2),
                    aggr=aggr,
                )
            )
        layers.append(
            NNConv(
                hidden_channels,
                out_channels,
                nn=nn.LazyLinear(hidden_channels * out_channels),
                aggr=aggr,
            )
        )
    elif gnn_type == "SGC":
        layers.append(SGConv(in_channels, hidden_channels, K=1))
        layers.append(SGConv(hidden_channels, hidden_channels, K=num_layers - 2))
        layers.append(SGConv(hidden_channels, out_channels, K=1))
    elif gnn_type in _SIMPLE_TYPES:
        layers.append(gnn(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            layers.append(gnn(hidden_channels, hidden_channels))
        layers.append(gnn(hidden_channels, out_channels))
    return layers


__all__ = ["GNN_TYPES", "SumGNN", "build_gnn", "get_layer_type"]
