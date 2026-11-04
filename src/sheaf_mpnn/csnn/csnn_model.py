# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Cooperative Sheaf Neural Network (CSNN) model."""

from __future__ import annotations

from enum import Enum, auto

import torch.nn.functional as F
from torch import Tensor, nn

from sheaf_mpnn.csnn.csnn_layers import CSNNConv


class CSNNVariant(Enum):
    """Selects the restriction-map family used by the sheaf learners."""

    # Learner emits an extra channel decoded as a positive scale c, so the
    # restriction map is c * Q (a conformal map).
    CONFORMAL = auto()
    # c is fixed at 1, leaving plain orthogonal restriction maps.
    NON_CONFORMAL = auto()

    @property
    def conformal(self) -> bool:
        return self is CSNNVariant.CONFORMAL


class CSNNModel(nn.Module):
    """Cooperative Sheaf Neural Network for node-level prediction.

    Learns a *conformal* sheaf whose restriction maps carry both an orthogonal
    rotation and a positive scale, and diffuses along the incoming and outgoing
    sheaf Laplacians in sequence.  Each layer combines the diffusion term with a
    learnable per-stalk skip, giving the cooperative update
    ``x <- (1 + tanh(eps)) * x0 - L x``.

    Follows the same encoder -> layers -> decoder pipeline as ``NSDModel``.
    Graph-level pooling is not part of the model: ``GCLightningModule`` replaces
    ``decoder`` with ``nn.Identity`` and pools the stalk features itself.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: CSNNVariant = CSNNVariant.CONFORMAL,
        orth_strategy: str = "householder",
        clamp_val: float = 10.0,
        left_weights: bool = True,
        right_weights: bool = True,
        use_act: bool = True,
        layer_norm: bool = False,
        batch_norm: bool = False,
        csnn_head_layers: int = 1,
        gnn_layers: int = 0,
        gnn_hidden: int = 16,
        gnn_default: int = 1,
        gnn_type: str = "SAGE",
        gnn_residual: bool = False,
        linear_emb: bool = False,
        pe_size: int = 0,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
    ):
        """Initializes a CSNN model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. num classes).
            stalk_dim: Stalk dimension ``d``. Must be greater than 1 -- a
                1-dimensional stalk admits no non-trivial orthogonal map.
            hidden_dim: Feature width per stalk entry. The encoded node state
                has size ``stalk_dim * hidden_dim``.
            num_layers: Number of cooperative sheaf layers. Must be positive.
            variant: CONFORMAL (scaled maps) or NON_CONFORMAL (orthogonal only).
            orth_strategy: Orthogonal parameterization. Defaults to
                ``"householder"``, the reference CSNN choice -- note this
                differs from the ``"cayley"`` default used elsewhere here.
            clamp_val: Clamping bound for the Cayley transform.
            left_weights: Learn a ``d x d`` stalk-mixing transform per layer.
            right_weights: Learn a channel-mixing transform per layer.
            use_act: Apply GELU after the encoder and to each diffusion term.
            layer_norm: Apply ``LayerNorm`` after each layer.
            batch_norm: Apply ``BatchNorm1d`` after each layer.
            csnn_head_layers: Depth of the classification MLP. ``1`` means a
                single output projection. Values above 1 prepend that many minus
                one GELU-separated hidden layers. (This is the reference's
                ``--num_heads``; it is unrelated to attention heads.)
            gnn_layers: Sheaf-learner backbone depth. ``0`` selects an MLP.
            gnn_hidden: Sheaf-learner hidden width. Used even when
                ``gnn_layers == 0``, where it sizes the MLP.
            gnn_default: Sheaf-learner preset -- ``1`` and ``2`` are the two
                reference ``GraphSAGE`` configurations, ``0`` builds a custom
                stack from ``gnn_type``.
            gnn_type: Convolution type when ``gnn_default == 0``.
            gnn_residual: Residual connections in the custom learner stack.
            linear_emb: Linear embed/project wrapper for the custom stack.
            pe_size: Width of the optional positional encoding passed to
                ``forward``. Must match the ``pe`` tensor's width.
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on stalk features inside each layer.
            normalize_output: L2-normalise the representation before the
                decoder (Lv et al., 2021). Off by default, matching the
                reference CSNN.

        Raises:
            ValueError: If any dimension or depth argument is out of range.
        """
        super().__init__()
        if stalk_dim <= 1:
            raise ValueError("stalk_dim must be greater than 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one CSNN layer")
        if csnn_head_layers <= 0:
            raise ValueError("csnn_head_layers must be positive")
        if pe_size < 0:
            raise ValueError("pe_size must be non-negative")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pe_size = pe_size
        self.use_act = use_act
        self.normalize_output = normalize_output
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        self.layers = nn.ModuleList(
            [
                CSNNConv(
                    stalk_dim=stalk_dim,
                    hidden_dim=hidden_dim,
                    variant_conformal=variant.conformal,
                    orth_strategy=orth_strategy,
                    clamp_val=clamp_val,
                    left_weights=left_weights,
                    right_weights=right_weights,
                    use_act=use_act,
                    dropout=dropout,
                    gnn_layers=gnn_layers,
                    gnn_hidden=gnn_hidden,
                    gnn_default=gnn_default,
                    gnn_type=gnn_type,
                    gnn_residual=gnn_residual,
                    linear_emb=linear_emb,
                    pe_size=pe_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(context_dim) for _ in range(num_layers)])
            if layer_norm
            else None
        )
        self.batch_norms = (
            nn.ModuleList(
                [nn.BatchNorm1d(context_dim, affine=True) for _ in range(num_layers)]
            )
            if batch_norm
            else None
        )

        # The hidden MLP stages live in `head` so that swapping `decoder` for
        # nn.Identity (graph classification) drops only the output projection.
        # Under that swap the head runs per-node, before pooling, whereas the
        # reference pools first -- identical while csnn_head_layers == 1 (the
        # Identity head), which is what every shipped preset uses.
        head: list[nn.Module] = []
        for _ in range(csnn_head_layers - 1):
            head.append(nn.Linear(context_dim, context_dim))
            head.append(nn.GELU())
        self.head: nn.Module = nn.Sequential(*head) if head else nn.Identity()
        self.decoder: nn.Module = nn.Linear(context_dim, out_channels)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, CSNNConv)
            layer.reset_parameters()
        for norms in (self.layer_norms, self.batch_norms):
            if norms is not None:
                for norm in norms:
                    assert isinstance(norm, nn.LayerNorm | nn.BatchNorm1d)
                    norm.reset_parameters()
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        # `decoder` may have been replaced with nn.Identity for graph-level
        # pooling, which has no parameters to reset.
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
    ) -> Tensor:
        """Runs the CSNN encoder, cooperative sheaf layers, and decoder.

        Args:
            x: Raw node features ``[num_nodes, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            edge_attr: Optional edge features, forwarded to the sheaf learners.
                Only consumed when ``gnn_default == 0`` and the chosen
                ``gnn_type`` accepts edge attributes.
            pe: Optional positional encoding ``[num_nodes, pe_size]``,
                concatenated to the sheaf-learner input.

        Returns:
            Node outputs ``[num_nodes, out_channels]``, or the flat stalk
            representation ``[num_nodes, stalk_dim * hidden_dim]`` when
            ``decoder`` has been replaced with ``nn.Identity``.
        """
        if pe is not None and pe.size(-1) != self.pe_size:
            raise ValueError(
                f"pe has width {pe.size(-1)} but pe_size is {self.pe_size}"
            )

        num_nodes = x.size(0)
        x = self.encoder(self.input_dropout_layer(x))
        if self.use_act:
            x = F.gelu(x)

        x_stalk = x.view(num_nodes, self.stalk_dim, self.hidden_dim)
        for i, layer in enumerate(self.layers):
            # Drop once: the sheaf learners and the layer's skip term must see
            # the same mask, as in the reference implementation.
            x_flat = self.dropout_layer(x_stalk.reshape(num_nodes, -1))
            x_stalk = x_flat.view(num_nodes, self.stalk_dim, self.hidden_dim)
            out = layer(x_flat, x_stalk, edge_index, edge_attr, pe)
            if self.layer_norms is not None:
                out = self.layer_norms[i](out)
            if self.batch_norms is not None:
                out = self.batch_norms[i](out)
            x_stalk = out.view(num_nodes, self.stalk_dim, self.hidden_dim)

        x_out = x_stalk.reshape(num_nodes, -1)
        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)
        return self.decoder(self.head(x_out))


__all__ = ["CSNNModel", "CSNNVariant"]
