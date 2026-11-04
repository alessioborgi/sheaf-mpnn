# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.nsd.poly_nsd_layers import (
    BasePolyHetNSDConv,
    DiagonalPolyHetNSDConv,
    GeneralPolyHetNSDConv,
    LowRankPolyHetNSDConv,
    OrthogonalPolyHetNSDConv,
)


class PolyHetNSDVariant(Enum):
    """Restriction-map family for the polynomial het-NSD model."""

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            PolyHetNSDVariant.DIAGONAL: DiagonalPolyHetNSDConv,
            PolyHetNSDVariant.GENERAL: GeneralPolyHetNSDConv,
            PolyHetNSDVariant.ORTHOGONAL: OrthogonalPolyHetNSDConv,
            PolyHetNSDVariant.GENERAL_ATTENTION: GeneralPolyHetNSDConv,
            PolyHetNSDVariant.ORTHOGONAL_ATTENTION: OrthogonalPolyHetNSDConv,
            PolyHetNSDVariant.LOW_RANK: LowRankPolyHetNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {PolyHetNSDVariant.DIAGONAL, PolyHetNSDVariant.LOW_RANK}:
            return {}
        if self in {
            PolyHetNSDVariant.GENERAL_ATTENTION,
            PolyHetNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}


class PolyHetNSDModel(nn.Module):
    """Polynomial Heterogeneous Neural Sheaf Diffusion (PolyHetNSD) model.

    Combines the polynomial spectral filter from PolyNSD with the het-graph context
    (node/edge type information) from HetNSD.  Each diffusion layer applies a
    degree-K Chebyshev polynomial of the normalised sheaf Laplacian, with restriction
    maps conditioned on node/edge type information.

    Follows the same encoder -> layers -> decoder pipeline as HetNSDModel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: PolyHetNSDVariant = PolyHetNSDVariant.GENERAL,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_self_loops: bool = False,
        orth_strategy: str = "cayley",
        rank: int = 1,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
    ):
        """Initialises a PolyHetNSD model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node.
            stalk_dim: Stalk dimension d.
            hidden_dim: Feature width f inside each stalk channel.
            num_layers: Number of polynomial het-NSD convolution layers.
            variant: Restriction-map family (PolyHetNSDVariant).
            alpha: Diffusion step size passed to each layer constructor.
            degree: Polynomial filter degree K.
            lambda_max: Spectral upper bound of the normalised sheaf Laplacian.
            hp_alpha_init: Initial value of the per-layer learnable HP skip weight.
            num_node_types: Number of node types (required when use_node_types=True).
            num_edge_types: Number of edge types (required when use_edge_types=True).
            use_features: Include node features in restriction-map MLP context.
            use_node_types: Include node-type one-hots in MLP context.
            use_edge_types: Include edge-type one-hots in MLP context.
            ensemble: Use independent MLPs per edge type (TypeEnsemble).
            add_self_loops: If True, add self-loops before degree normalisation.
            orth_strategy: "cayley" or "fasth" for the ORTHOGONAL variant.
            rank: Low-rank map rank r for the LOW_RANK variant.
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on stalk features between layers.
            normalize_output: If True, L2-normalise representation before the decoder.
            jknet: If True, concatenate hidden states from all layers before decoder.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one PolyHetNSD layer")
        if degree < 1:
            raise ValueError("degree must be at least 1")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        extra_kwargs = variant.layer_kwargs.copy()
        if variant in {
            PolyHetNSDVariant.ORTHOGONAL,
            PolyHetNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            extra_kwargs["orth_strategy"] = orth_strategy
        if variant == PolyHetNSDVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_self_loops=add_self_loops,
                    degree=degree,
                    lambda_max=lambda_max,
                    hp_alpha_init=hp_alpha_init,
                    **extra_kwargs,
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
            assert isinstance(layer, BasePolyHetNSDConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, polynomial het-NSD diffusion layers, and decoder.

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


__all__ = ["PolyHetNSDVariant", "PolyHetNSDModel"]
