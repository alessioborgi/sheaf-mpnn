# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.sheaf_adj.sheaf_adj_layers import (
    BaseSheafAdjConv,
    DiagonalSheafAdjConv,
    GeneralSheafAdjConv,
    LowRankSheafAdjConv,
    OrthogonalSheafAdjConv,
)


class SheafAdjVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            SheafAdjVariant.DIAGONAL: DiagonalSheafAdjConv,
            SheafAdjVariant.GENERAL: GeneralSheafAdjConv,
            SheafAdjVariant.ORTHOGONAL: OrthogonalSheafAdjConv,
            SheafAdjVariant.LOW_RANK: LowRankSheafAdjConv,
            SheafAdjVariant.GENERAL_ATTENTION: GeneralSheafAdjConv,
            SheafAdjVariant.ORTHOGONAL_ATTENTION: OrthogonalSheafAdjConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        if self in {SheafAdjVariant.DIAGONAL, SheafAdjVariant.LOW_RANK}:
            return {}
        if self in {
            SheafAdjVariant.GENERAL_ATTENTION,
            SheafAdjVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
        use_edge_weights: bool = False,
    ) -> dict[str, Any]:
        kwargs = self.layer_kwargs.copy()
        if self in {SheafAdjVariant.ORTHOGONAL, SheafAdjVariant.ORTHOGONAL_ATTENTION}:
            kwargs["orth_strategy"] = orth_strategy
            kwargs["use_edge_weights"] = use_edge_weights
        if self == SheafAdjVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


class SheafAdjModel(nn.Module):
    """End-to-end SheafAdjacency model.

    Lifts raw node features into stalk space, applies a stack of SheafAdjacency
    convolution layers, and decodes the flattened stalk representation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: SheafAdjVariant = SheafAdjVariant.GENERAL,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
        add_eps: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        second_linear: bool = False,
        use_edge_weights: bool = False,
        learn_alpha: bool = True,
    ):
        """Initializes a SheafAdjacency model for node-level prediction.

        Args:
            in_channels (int): Number of raw input features per node.
            out_channels (int): Number of output channels per node (e.g. num classes).
            stalk_dim (int, optional): Stalk dimension. Each node is represented
                internally as a matrix with shape ``[stalk_dim, hidden_dim]``.
                Defaults to 4.
            hidden_dim (int, optional): Feature dimension inside each stalk channel.
                The encoded node state has size ``d * hidden_dim``. Defaults to 16.
            num_layers (int, optional): Number of SheafAdjacency convolution layers.
                Must be positive. Defaults to 2.
            variant (SheafAdjVariant, optional): Restriction-map family.
                ``DIAGONAL`` is cheapest, ``GENERAL`` is most expressive,
                ``ORTHOGONAL`` uses orthogonal maps. Defaults to ``GENERAL``.
            alpha (float, optional): Initial learnable diffusion step size per layer.
                Defaults to 1.0.
            add_self_loops (bool, optional): If ``True``, self-loops are added before
                computing degree normalisation in each layer. Defaults to ``True``.
            orth_strategy (str, optional): Orthogonality strategy for the
                ``ORTHOGONAL`` variant: ``"cayley"`` or ``"fasth"``. Defaults to
                ``"cayley"``.
            rank (int, optional): Rank of each restriction map for the ``LOW_RANK``
                variant. Must be positive. Ignored for other variants. Defaults to 1.
            add_eps (bool, optional): If ``True``, each layer applies a learnable
                per-stalk rescaling ``eps in R^d`` to the skip term (Remark 4).
                Defaults to ``False``.
            input_dropout (float, optional): Dropout probability applied to raw
                input features before encoding. Defaults to 0.0.
            dropout (float, optional): Dropout probability applied to stalk features
                between layers. Defaults to 0.0.
            normalize_output (bool, optional): If ``True``, L2-normalise the
                representation before the decoder (Lv et al., 2021). Defaults to
                ``True``.
            jknet (bool, optional): If ``True``, collect hidden states from every
                layer and concatenate them before the decoder (Xu et al., 2018).
                Defaults to ``False``.
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
            learn_alpha: If False, alpha stays fixed at its init value
                (Bodnar-exact update). Defaults to True.
            second_linear: Adds a second encoder linear stage (reference
                ``--second_linear``). Defaults to False.
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant). Defaults to False.
            use_edge_weights: Learned symmetric per-edge scalars rescaling
                orthogonal maps (reference edge_weights). Defaults to False.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one SheafAdj layer")

        self.stalk_dim = stalk_dim
        # Fixed lp/hp channels extend the stalk beyond the learned dims.
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = self.final_d * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)
        # Second encoder stage (reference second_linear).
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        extra_kwargs = variant.build_kwargs(
            orth_strategy=orth_strategy,
            rank=rank,
            use_edge_weights=use_edge_weights,
        )

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    sparse_learner=sparse_learner,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        if not learn_alpha:
            # Fixed step size on the adjacency term (Bodnar-style alpha=const).
            for layer in self.layers:
                layer.alpha.requires_grad_(False)

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseSheafAdjConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the SheafAdj encoder, propagation layers, and decoder.

        Args:
            x (torch.Tensor): Raw node features [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].

        Returns:
            torch.Tensor: Node outputs [num_nodes, out_channels].
        """
        if x.dim() != 2 or x.size(1) != self.encoder.in_features:
            raise ValueError(
                f"x must have shape [num_nodes, {self.encoder.in_features}],"
                f" got {tuple(x.shape)}"
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges],"
                f" got {tuple(edge_index.shape)}"
            )

        x_stalk = self.encoder(self.input_dropout_layer(x))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)

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
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["SheafAdjVariant", "SheafAdjModel"]
