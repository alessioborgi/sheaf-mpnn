# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.sheaf_attn.sheaf_attn_layers import SheafAttnConv


class SheafAttnVariant(Enum):
    """Selects between the two SheafAN parameterisations from the paper."""

    SHEAFAN = auto()  # Eq. (5)  No residual connection.
    RES_SHEAFAN = auto()  # Eq. (6)  Residual connection (high/low-pass filter).

    @property
    def residual(self) -> bool:
        return self == SheafAttnVariant.RES_SHEAFAN


class SheafAttnModel(nn.Module):
    """Sheaf Attention Network (SheafAN / Res-SheafAN).

    Generalises GAT to cellular sheaves by pairing a GAT-style attention
    mechanism with learned orthogonal restriction maps (Barbero et al., 2022).

    Each layer computes either the SheafAN update (Eq. 5) or the Res-SheafAN
    update (Eq. 6).

    Follows the same encoder -> layers -> decoder pipeline as NSDModel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: SheafAttnVariant = SheafAttnVariant.SHEAFAN,
        num_heads: int = 1,
        leaky_relu_slope: float = 0.2,
        clamp_val: float = 10.0,
        orth_strategy: str = "cayley",
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        add_eps: bool = False,
        sparse_learner: bool = False,
        second_linear: bool = False,
    ):
        """Initializes a SheafAttn model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. num classes).
            stalk_dim: Stalk dimension. Each node is represented internally as a
                matrix with shape ``[stalk_dim, hidden_dim]``.
            hidden_dim: Feature dimension inside each stalk channel. The encoded
                node state has size ``stalk_dim * hidden_dim``.
            num_layers: Number of SheafAttn convolution layers. Must be positive.
            variant: SHEAFAN (no residual) or RES_SHEAFAN (residual connection).
            num_heads: Independent attention heads per layer (averaged).
            leaky_relu_slope: Negative slope for LeakyReLU in attention logits.
            clamp_val: Clamping bound for Cayley transform inputs (ignored for fasth).
            orth_strategy: "cayley" or "fasth" (Householder). Defaults to "cayley".
            input_dropout: Dropout probability applied to raw input features before
                encoding. Defaults to 0.0.
            dropout: Dropout probability applied to stalk features between layers.
                Defaults to 0.0.
            normalize_output: If ``True``, L2-normalise the representation before the
                decoder (Lv et al., 2021). If ``jknet`` is ``True``, each layer's
                output is also normalised before concatenation. Defaults to ``True``.
            jknet: If ``True``, concatenate hidden states from every layer before
                the decoder (Xu et al., 2018). Normalization is controlled by
                ``normalize_output``. Intended for link prediction. Defaults to
                ``False``.
            add_eps: If ``True`` and ``variant`` is ``RES_SHEAFAN``, each layer
                applies a learnable per-stalk rescaling ``eps in R^d`` to the skip
                term (Remark 4). Has no effect for plain ``SHEAFAN``. Defaults to
                ``False``.
            second_linear: Adds a second encoder linear stage (reference
                ``--second_linear``). Defaults to False.
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one SheafAttn layer")
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
        # Second encoder stage (reference second_linear).
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        self.layers = nn.ModuleList(
            [
                SheafAttnConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    leaky_relu_slope=leaky_relu_slope,
                    context_dim=context_dim,
                    clamp_val=clamp_val,
                    residual=variant.residual,
                    orth_strategy=orth_strategy,
                    add_eps=add_eps,
                    sparse_learner=sparse_learner,
                )
                for _ in range(num_layers)
            ]
        )

        # JKNet expands the decoder input to L  context_dim (Xu et al., 2018).
        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, SheafAttnConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the SheafAttn encoder, attention layers, and decoder.

        Args:
            x (torch.Tensor): Raw node features [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Node outputs [num_nodes, out_channels].
        """
        # Lift raw features to stalk space:
        # [N, in_channels] -> [N, stalk_dim, hidden_dim].
        x_stalk = self.encoder(self.input_dropout_layer(x))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.stalk_dim, self.hidden_dim)

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
            # L2-normalise final representation - Lv et al. (2021).
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["SheafAttnVariant", "SheafAttnModel"]
