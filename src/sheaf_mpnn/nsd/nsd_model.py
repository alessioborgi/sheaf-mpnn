# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

from enum import Enum, auto
from typing import Any

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.nsd.nsd_layers import (
    BaseNSDConv,
    DiagonalNSDConv,
    GeneralNSDConv,
    LowRankNSDConv,
    OrthogonalNSDConv,
)


class NSDVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            NSDVariant.DIAGONAL: DiagonalNSDConv,
            NSDVariant.GENERAL: GeneralNSDConv,
            NSDVariant.ORTHOGONAL: OrthogonalNSDConv,
            NSDVariant.LOW_RANK: LowRankNSDConv,
            NSDVariant.GENERAL_ATTENTION: GeneralNSDConv,
            NSDVariant.ORTHOGONAL_ATTENTION: OrthogonalNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {NSDVariant.DIAGONAL, NSDVariant.LOW_RANK}:
            return {}
        if self in {NSDVariant.GENERAL_ATTENTION, NSDVariant.ORTHOGONAL_ATTENTION}:
            return {"use_attention": True}
        return {"use_attention": False}


class NSDModel(nn.Module):
    """End-to-end Neural Sheaf Diffusion (NSD) model.

    The wrapper lifts raw node features into stalk features, applies a stack of NSD
    convolution layers, and decodes the flattened stalk representation back to the
    requested output dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: NSDVariant = NSDVariant.GENERAL,
        alpha: float = 1.0,
        add_self_loops: bool = True,
        orth_strategy: str = "cayley",
        rank: int = 1,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
    ):
        """Initializes an NSD model for node-level prediction.

        Args:
            in_channels (int): Number of raw input features per node.
            out_channels (int): Number of output channels per node (e.g. num classes).
            stalk_dim (int, optional): Stalk dimension. Each node is represented
                internally as a matrix with shape ``[stalk_dim, hidden_dim]``.
            hidden_dim (int, optional): Feature dimension inside each stalk channel.
                The encoded node state has size ``d * hidden_dim``.
            num_layers (int, optional): Number of NSD convolution layers. Must be
                positive.
            variant (NSDVariant, optional): Restriction-map family. ``DIAGONAL`` is
                cheapest, ``GENERAL`` is most expressive, ``ORTHOGONAL`` uses orthogonal
                maps (via Cayley or Householder parameterisation). ``GENERAL_ATTENTION``
                and ``ORTHOGONAL_ATTENTION`` use an attention-based map initialisation.
            alpha (float, optional): Initial learnable diffusion step size per layer.
            add_self_loops (bool, optional): If ``True``, self-loops are added to the
                graph before computing degree normalization in each layer. Defaults to
                ``True``.
            orth_strategy (str, optional): Orthogonality strategy for the
                ``ORTHOGONAL`` variant: "cayley" or "fasth". Defaults to "cayley".
            rank (int, optional): Rank of each restriction map for the ``LOW_RANK``
                variant. Must be positive. Ignored for other variants. Defaults to 1.
            input_dropout (float, optional): Dropout probability applied to raw
                input features before encoding. Defaults to 0.0.
            dropout (float, optional): Dropout probability applied to stalk features
                between layers. Defaults to 0.0.
            normalize_output (bool, optional): If ``True``, L2-normalise the
                representation before the decoder (Lv et al., 2021). If ``jknet``
                is ``True``, each layer's output is also normalised before
                concatenation. Defaults to ``True``.
            jknet (bool, optional): If ``True``, collect hidden states from every
                layer and concatenate them before the decoder (Xu et al., 2018).
                Normalization is controlled by ``normalize_output``. Intended for
                link prediction. Defaults to ``False``.

        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one NSD layer")

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

        extra_kwargs = variant.layer_kwargs.copy()
        if variant == NSDVariant.ORTHOGONAL:
            extra_kwargs["orth_strategy"] = orth_strategy
        if variant == NSDVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,  # 'f' for W2 [f x f]
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,  # 'd*f' for MLP input [2*df x hidden]
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        # JKNet expands decoder input to L  context_dim (Xu et al., 2018).
        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseNSDConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the NSD encoder, diffusion layers, and decoder.

        Args:
            x (torch.Tensor): Raw node features with shape
                ``[num_nodes, in_channels]``.
            edge_index (torch.Tensor): Graph connectivity in COO format with shape
                ``[2, num_edges]``.

        Returns:
            torch.Tensor: Node outputs with shape ``[num_nodes, out_channels]``.
        """
        # Lift raw features to stalk space: [N, in_channels] -> [N, d, f].
        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []
        for layer in self.layers:
            # Flatten stalk to [N, d*f] as context for restriction-map generation.
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
            # L2-normalise final representation — Lv et al. (2021).
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["NSDVariant", "NSDModel"]
