# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous connection-Laplacian NSD model."""

from __future__ import annotations

import torch

from sheaf_mpnn.nsd.conn_nsd_model import ConnectionLaplacianNSDModel


class HetConnectionLaplacianNSDModel(ConnectionLaplacianNSDModel):
    """Heterogeneous connection-Laplacian NSD (Barbero et al., 2022) model.

    Identical to
    :class:`~sheaf_mpnn.nsd.conn_nsd_model.ConnectionLaplacianNSDModel`: the
    orthogonal transport maps are derived once from the raw node features by
    local PCA + SVD alignment, with no map generator. Like
    :class:`~hetsheaf_mpnn.id_sheaf.id_sheaf_model.HetIdentitySheafModel`, it
    therefore has no het-context flags: ``node_types`` and ``edge_types`` are
    accepted by ``forward`` for het-pipeline parity and ignored. Edge types
    could not be honoured anyway, since the map cache symmetrises and
    KNN-augments the graph, so the layers see edges that carry no type.

    ``to_homogeneous()`` zero-pads every node type's features to a shared
    width, so the local PCA runs on that padded space. A dataset whose nodes
    carry no features (e.g. Freebase, flattened to a constant column) fails
    the ``in_channels >= stalk_dim`` check at construction.
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the connection-Laplacian NSD model on a flattened het graph.

        Args:
            x: Raw node features ``[N, in_channels]``.
            edge_index: Graph connectivity ``[2, E]``.
            node_types: Node-type indices ``[N]``. Accepted for het-pipeline
                parity; unused (feature-derived maps).
            edge_types: Edge-type indices ``[E]``. Accepted for het-pipeline
                parity; unused (feature-derived maps).

        Returns:
            Node-level outputs ``[N, out_channels]``.
        """
        return super().forward(x, edge_index)


__all__ = ["HetConnectionLaplacianNSDModel"]
