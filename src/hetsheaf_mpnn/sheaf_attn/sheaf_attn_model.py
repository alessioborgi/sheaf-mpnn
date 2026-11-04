# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.sheaf_attn.sheaf_attn_layers import HetSheafAttnConv


class HetSheafAttnVariant(Enum):
    SHEAFAN = auto()
    RES_SHEAFAN = auto()

    @property
    def residual(self) -> bool:
        return self == HetSheafAttnVariant.RES_SHEAFAN


class HetSheafAttnModel(nn.Module):
    """Heterogeneous Sheaf Attention Network (HetSheafAN / HetRes-SheafAN).

    Type information is incorporated into both attention scores (via node-type
    one-hots) and restriction-map generators (via node/edge-type one-hots),
    giving the model full awareness of graph heterogeneity.

    Follows the same encoder -> layers -> decoder pipeline as SheafAttnModel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetSheafAttnVariant = HetSheafAttnVariant.SHEAFAN,
        num_heads: int = 1,
        leaky_relu_slope: float = 0.2,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        clamp_val: float = 10.0,
        orth_strategy: str = "cayley",
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
            raise ValueError("must have at least one HetSheafAttn layer")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        self.layers = nn.ModuleList(
            [
                HetSheafAttnConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    leaky_relu_slope=leaky_relu_slope,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    clamp_val=clamp_val,
                    residual=variant.residual,
                    orth_strategy=orth_strategy,
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

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetSheafAttnConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het-SheafAttn layers, and decoder.

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


__all__ = ["HetSheafAttnVariant", "HetSheafAttnModel"]
