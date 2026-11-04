# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
from torch import nn

from sheaf_mpnn.base_conv import BaseSheafConv


class DNSDConv(nn.Module):
    r"""Deep Neural Sheaf Diffusion convolution (Bourgerie et al., 2026).

    Wraps a single :class:`~sheaf_mpnn.sheaf_adj.sheaf_adj_layers.BaseSheafAdjConv`
    instance, composed via its ``residual=False`` bare-message extension
    point, reusing the restriction-map and normalization logic. The wrapped
    layer must already be constructed with
    ``residual=False`` (set as a post-construction attribute, since
    ``residual`` is not a ``BaseSheafAdjConv`` constructor parameter) so its
    ``forward()`` returns the bare, activated diffusion term
    :math:`\alpha\,\sigma(\hat{A}_\mathcal{F}\mathbf{z})` with no skip added.

    This module adds exactly the two per-layer DNSD deltas that are *not*
    already covered by the wrapped layer's own machinery:

    * ``gate``: a per-(node, stalk) sigmoid gate, computed from the
      pre-layer stalk state alone, multiplied onto the bare diffusion term
      before the skip is added (Eq. 5, Table 4).
    * the skip connection itself, added *outside* the wrapped layer (since
      ``residual=False`` suppressed the layer's own skip) -- identical in
      shape to :class:`BaseSheafAdjConv`'s own skip logic.

    ``odd`` (tanh instead of the inherited ``ELU`` activation) is not a
    property of this wrapper: it is applied by overwriting the wrapped
    layer's ``sigma`` attribute at construction time (see
    :func:`sheaf_mpnn.dnsd.dnsd_model.DNSDModel`), the same pattern
    :class:`~sheaf_mpnn.sheaf_attn.sheaf_attn_layers.SheafAttnConv` already
    uses to override its inherited activation.

    LayerNorm (the third DNSD delta) is applied at the model level, once per
    layer, after this module's ``forward()`` returns -- not here, matching
    the reference implementation's ``layer_norm_mode="embedding"`` (applied
    to the full per-node embedding, not inside the per-layer conv).
    """

    def __init__(
        self,
        sheaf_adj_layer: BaseSheafConv,
        stalk_dim: int,
        in_channels: int,
        gate: bool = False,
        add_eps: bool = False,
    ):
        """Initializes a DNSD convolution wrapping one SheafAdjacency layer.

        Args:
            sheaf_adj_layer: A ``BaseSheafAdjConv`` (or, for
                :class:`~hetsheaf_mpnn.dnsd.dnsd_layers.HetDNSDConv`,
                ``BaseHetSheafAdjConv``) subclass instance, already
                constructed with ``residual=False`` set as a
                post-construction attribute by the caller.
            stalk_dim: Stalk dimension d (must match ``sheaf_adj_layer``'s).
            in_channels: Feature dimension f inside each stalk channel (must
                match ``sheaf_adj_layer``'s).
            gate: If True, applies a learnable per-(node, stalk) sigmoid gate
                to the diffusion term before the skip is added. The gate is
                computed from ``x_stalk`` alone (the pre-layer stalk state).
                The gate linear map is shared across stalks (its weight
                shape does not depend on ``stalk_dim``); each stalk still
                gets its own gate value because it sees its own slice of the
                input. Zero-initialised so every gate starts at exactly 0.5.
            add_eps: If True, adds a learnable per-stalk rescaling
                ``eps in R^d`` to the skip term, matching
                ``BaseSheafAdjConv``'s own ``add_eps`` option (duplicated
                here since the wrapped layer's own skip -- and thus its own
                ``eps`` -- is bypassed by ``residual=False``).
        """
        super().__init__()
        if sheaf_adj_layer.residual:
            raise ValueError(
                "sheaf_adj_layer must be constructed with residual=False "
                "(set post-construction, e.g. `layer.residual = False`) so "
                "DNSDConv owns the single skip connection, not both."
            )
        self.sheaf_adj_layer = sheaf_adj_layer
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels
        self.gate = gate
        self.add_eps = add_eps

        if gate:
            self.gate_lin = nn.Linear(in_channels, 1, bias=True)
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)

        if add_eps:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

    def reset_parameters(self):
        self.sheaf_adj_layer.reset_parameters()
        if self.gate:
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)
        if self.add_eps:
            nn.init.zeros_(self.eps)

    def forward(
        self, x_feat: torch.Tensor, x_stalk: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Applies one DNSD propagation step to lifted node features.

        Args:
            x_feat: Node context features [num_nodes, context_dim].
            x_stalk: Lifted node features [num_nodes, stalk_dim, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Updated stalk features [num_nodes, stalk_dim, in_channels].
        """
        # Bare, activated diffusion term: alpha * sigma(A_hat_F @ z), no skip
        # (the wrapped layer's residual=False suppresses its own skip+msg).
        msg = self.sheaf_adj_layer(x_feat, x_stalk, edge_index)
        return self._gated_skip(msg, x_stalk)

    def _gated_skip(self, msg: torch.Tensor, x_stalk: torch.Tensor) -> torch.Tensor:
        """Gates the bare diffusion term and adds the skip connection.

        Args:
            msg: Bare diffusion term from the wrapped layer [N, d, f].
            x_stalk: Pre-layer stalk state [N, d, f].

        Returns:
            torch.Tensor: Updated stalk features [N, d, f].
        """
        if self.gate:
            n, d = x_stalk.shape[0], x_stalk.shape[1]
            gate_val = torch.sigmoid(
                self.gate_lin(x_stalk.reshape(n * d, self.in_channels))
            ).view(n, d, 1)
            msg = gate_val * msg

        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip + msg


__all__ = ["DNSDConv"]
