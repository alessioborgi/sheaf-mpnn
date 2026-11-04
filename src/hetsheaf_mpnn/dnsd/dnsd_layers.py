# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch

from sheaf_mpnn.dnsd.dnsd_layers import DNSDConv


class HetDNSDConv(DNSDConv):
    """Heterogeneous Deep Neural Sheaf Diffusion convolution.

    Identical to :class:`~sheaf_mpnn.dnsd.dnsd_layers.DNSDConv` (same gate and
    skip connection), but wraps a ``residual=False``
    :class:`~hetsheaf_mpnn.sheaf_adj.sheaf_adj_layers.BaseHetSheafAdjConv` and
    forwards ``node_types`` / ``edge_types`` to it, so the het context reaches
    the restriction-map generator. The gate stays type-blind: it is computed
    from the pre-layer stalk state alone, exactly as in the homogeneous layer.
    """

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one het-DNSD propagation step to lifted node features.

        Args:
            x_feat:     [N, context_dim]             Flattened stalk (context for maps).
            x_stalk:    [N, stalk_dim, in_channels]  Structured stalk to update.
            edge_index: [2, E]                       Graph connectivity.
            node_types: [N] LongTensor or None       Node-type indices.
            edge_types: [E] LongTensor or None       Edge-type indices.

        Returns:
            Updated stalk [N, stalk_dim, in_channels].
        """
        msg = self.sheaf_adj_layer(x_feat, x_stalk, edge_index, node_types, edge_types)
        return self._gated_skip(msg, x_stalk)


__all__ = ["HetDNSDConv"]
