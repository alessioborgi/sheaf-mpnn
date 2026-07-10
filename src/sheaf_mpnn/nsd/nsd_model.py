# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

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

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
        use_edge_weights: bool = False,
    ) -> dict[str, Any]:
        kwargs = self.layer_kwargs.copy()
        if self in {NSDVariant.ORTHOGONAL, NSDVariant.ORTHOGONAL_ATTENTION}:
            kwargs["orth_strategy"] = orth_strategy
            kwargs["use_edge_weights"] = use_edge_weights
        if self == NSDVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


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
        add_self_loops: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
        add_eps: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        second_linear: bool = False,
        use_edge_weights: bool = False,
        learn_alpha: bool = True,
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
            alpha (float, optional): Initial diffusion step size per layer.
                Defaults to 1.0, matching the reference update at init.
            learn_alpha (bool, optional): If ``False``, alpha stays fixed at
                its initial value (Bodnar-exact update: no step-size
                parameter); if ``True`` it is learned. Defaults to ``True``.
            add_self_loops (bool, optional): If ``True``, self-loops are added to the
                graph before computing degree normalization in each layer. Defaults
                to ``False``: the +1/+I degree augmentation now lives inside the
                normalization itself, so adding self-loops would augment twice.
            orth_strategy (str, optional): Orthogonality strategy for the
                ``ORTHOGONAL`` variant: "cayley", "fasth", "fasthpp", or
                "householder" (reference orgqr; fasthpp computes the same map
                natively). Defaults to "cayley".
            rank (int, optional): Rank of each restriction map for the ``LOW_RANK``
                variant. Must be positive. Ignored for other variants. Defaults to 1.
            add_eps (bool, optional): If ``True``, each layer applies a learnable
                per-stalk rescaling ``eps in R^d`` to the skip term (Remark 4).
                Defaults to ``True`` (reference parity: the original NSD always
                gates the skip with ``1 + tanh(eps)``, eps initialized at zero).
            input_dropout (float, optional): Dropout probability applied to raw
                input features before encoding. Defaults to 0.0.
            dropout (float, optional): Dropout probability applied to stalk features
                between layers. Defaults to 0.0.
            normalize_output (bool, optional): If ``True``, L2-normalise the
                representation before the decoder (Lv et al., 2021). If ``jknet``
                is ``True``, each layer's output is also normalised before
                concatenation. Defaults to ``False`` (reference parity: the
                original NSD decodes the raw representation).
            jknet (bool, optional): If ``True``, collect hidden states from every
                layer and concatenate them before the decoder (Xu et al., 2018).
                Normalization is controlled by ``normalize_output``. Intended for
                link prediction. Defaults to ``False``.
            add_lp (bool, optional): Appends one fixed stalk channel diffusing
                with the signless Laplacian D+A (reference ``--add_lp``).
            add_hp (bool, optional): Appends one fixed stalk channel diffusing
                with the standard Laplacian D-A (reference ``--add_hp``).
            sparse_learner (bool, optional): Stalk-summed map-generator input
                (reference LocalConcatSheafLearnerVariant), shrinking the
                learner from 2*final_d*hidden to 2*hidden inputs.
            second_linear (bool, optional): Adds a second encoder linear after
                the ELU+dropout stage (reference ``--second_linear``).

            use_edge_weights: Learned symmetric per-edge scalars rescaling
                orthogonal maps (reference edge_weights). Defaults to False.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one NSD layer")

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
        # Reference second_linear: extra encoder stage after ELU+dropout.
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
                    in_channels=hidden_dim,  # 'f' for W2 [f x f]
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,  # 'final_d*f' learner context
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    dropout=dropout,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    sparse_learner=sparse_learner,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        if not learn_alpha:
            # Bodnar-exact update: the diffusion step size stays fixed.
            for layer in self.layers:
                layer.alpha.requires_grad_(False)

        # JKNet expands decoder input to L  context_dim (Xu et al., 2018).
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
            assert isinstance(layer, BaseNSDConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
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

        # Lift raw features to stalk space: [N, in_channels] -> [N, final_d, f].
        # Reference pipeline: lin1 -> ELU -> dropout [-> lin12] before diffusion.
        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)

        layer_reps = []
        for i, layer in enumerate(self.layers):
            # Flatten stalk to [N, d*f] as context for restriction-map generation.
            x_feat = x_stalk.reshape(x_stalk.size(0), -1)
            # Reference: the layer-0 learner sees the un-dropped features.
            if i > 0:
                x_feat = self.dropout_layer(x_feat)
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


__all__ = ["NSDVariant", "NSDModel"]
