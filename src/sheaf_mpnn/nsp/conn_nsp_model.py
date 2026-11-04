# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Connection-Laplacian NSP model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.nsp.nsp_layers import ConnectionLaplacianNSPConv
from sheaf_mpnn.utils.connection_laplacian import TransportMapCache


class ConnectionLaplacianNSPModel(nn.Module):
    """NSP model with data-derived connection-Laplacian restriction maps.

    The second-order NSP recurrence is implemented via state augmentation
    (augmented stalk ``[x_curr; x_hist]``).  Per-edge orthogonal transport
    maps are computed once per forward pass from raw node features and reused
    across all NSP layers.

    Args:
        in_channels: Number of raw input features per node (must be >= stalk_dim).
        out_channels: Number of output channels per node (e.g. num classes).
        stalk_dim: Stalk / tangent-space dimension ``d``.
        hidden_dim: Feature dimension inside each stalk channel ``f``.
        num_layers: Number of NSP diffusion layers.
        add_self_loops: Whether to add self-loops for degree normalisation.
        input_dropout: Dropout on raw input features before encoding.
        dropout: Dropout on stalk features between layers.
        normalize_output: L2-normalise the final representation before decoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        add_self_loops: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if in_channels < stalk_dim:
            raise ValueError(
                f"in_channels ({in_channels}) must be >= stalk_dim ({stalk_dim})"
            )
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.normalize_output = normalize_output
        self.add_self_loops = add_self_loops

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, stalk_dim * hidden_dim)
        self._map_cache = TransportMapCache()
        self.layers = nn.ModuleList(
            [
                ConnectionLaplacianNSPConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    add_self_loops=False,  # model owns all edge preprocessing
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(stalk_dim * hidden_dim, out_channels)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()  # ty: ignore[call-non-callable]
        self.decoder.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the connection-Laplacian NSP model.

        Args:
            x: Raw node features ``[N, in_channels]``.
            edge_index: Graph connectivity ``[2, E]``.

        Returns:
            Node-level outputs ``[N, out_channels]``.
        """
        num_nodes = x.size(0)

        # Symmetrise, degree-augment and self-loop the graph, then derive the
        # maps. The cache covers the edge preprocessing too: it is a pure
        # function of the inputs but costs a near-full KNN search whenever
        # stalk_dim exceeds the mean degree, so redoing it every forward pass
        # would dominate both runtime and peak memory.
        edge_index, maps = self._map_cache.prepare(
            x, edge_index, self.stalk_dim, self.add_self_loops
        )

        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            num_nodes, self.stalk_dim, self.hidden_dim
        )
        # Augment with zero history stalk: [N, d, f] -> [N, 2d, f].
        x_stalk = torch.cat([x_stalk, torch.zeros_like(x_stalk)], dim=1)

        for layer in self.layers:
            x_stalk = self.dropout_layer(x_stalk)
            # maps acts as x_feat; ConnectionLaplacianNSPConv.get_map_products
            # returns (identity, maps.T) without consulting x_feat further.
            x_stalk = layer(maps, x_stalk, edge_index)

        x_out = x_stalk[:, : self.stalk_dim, :].reshape(num_nodes, -1)
        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)
        return self.decoder(x_out)


__all__ = ["ConnectionLaplacianNSPModel"]
