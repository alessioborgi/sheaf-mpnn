# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.nsp.nsp_layers import (
    BaseNSPConv,
    DiagonalNSPConv,
    GeneralNSPConv,
    LowRankNSPConv,
    OrthogonalNSPConv,
)


class NSPVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        mapping = {
            NSPVariant.DIAGONAL: DiagonalNSPConv,
            NSPVariant.GENERAL: GeneralNSPConv,
            NSPVariant.ORTHOGONAL: OrthogonalNSPConv,
            NSPVariant.GENERAL_ATTENTION: GeneralNSPConv,
            NSPVariant.ORTHOGONAL_ATTENTION: OrthogonalNSPConv,
            NSPVariant.LOW_RANK: LowRankNSPConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        # Diagonal and low-rank maps have no attention variant.
        if self in {NSPVariant.DIAGONAL, NSPVariant.LOW_RANK}:
            return {}
        if self in {NSPVariant.GENERAL_ATTENTION, NSPVariant.ORTHOGONAL_ATTENTION}:
            return {"use_attention": True}
        return {"use_attention": False}


class NSPModel(nn.Module):
    """Neural Sheaf Propagation (NSP) model.

    Implements the second-order NSP recurrence as a first-order SheafMPNN via
    state augmentation: node stalks are doubled to [x_curr; x_hist] in
    R^(2d x f), while restriction maps and bilateral weights remain d-dimensional.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: NSPVariant = NSPVariant.GENERAL,
        add_self_loops: bool = False,
        orth_strategy: str = "cayley",
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
    ):
        """Initializes an NSP model for node-level prediction.

        Args:
            in_channels (int): Number of raw input features per node.
            out_channels (int): Number of output channels per node (e.g. num classes).
            stalk_dim (int, optional): Stalk dimension. Each node is represented
                internally as a matrix with shape ``[stalk_dim, hidden_dim]``.
            hidden_dim (int, optional): Feature dimension inside each stalk channel.
                The encoded node state has size ``d * hidden_dim``.
            num_layers (int, optional): Number of NSP convolution layers. Must be
                positive.
            variant (NSPVariant, optional): Restriction-map family. ``DIAGONAL`` is
                cheapest, ``GENERAL`` is most expressive, ``ORTHOGONAL`` uses orthogonal
                maps (via Cayley or Householder parameterisation). ``GENERAL_ATTENTION``
                and ``ORTHOGONAL_ATTENTION`` use an attention-based map initialisation.
                ``LOW_RANK`` uses low-rank restriction maps (rank bounded by ``rank``).
            add_self_loops (bool, optional): If ``True``, self-loops are added to the
                graph before computing degree normalization in each layer. Defaults to
                ``True``.
            orth_strategy (str, optional): Orthogonality strategy for the
                ``ORTHOGONAL`` variant: "cayley" or "fasth". Defaults to "cayley".
            rank (int, optional): Rank of each restriction map for the ``LOW_RANK``
                variant. Must be positive. Ignored for other variants. Defaults to 1.
            add_eps (bool, optional): If ``True``, each layer applies learnable
                per-stalk rescalings ``eps_curr, eps_hist in R^d`` to the second-order
                skip terms (Remark 6). Defaults to ``False``.
            input_dropout (float, optional): Dropout probability applied to raw
                input features before encoding. Defaults to 0.0.
            dropout (float, optional): Dropout probability applied to stalk features
                between layers. Defaults to 0.0.
            normalize_output (bool, optional): If ``True``, L2-normalise the
                representation before the decoder (Lv et al., 2021). When
                ``jknet=True``, normalisation is applied per-layer before
                concatenation; the final concatenated representation is **not**
                normalised again. Defaults to ``True``.
            jknet (bool, optional): If ``True``, concatenate hidden states from
                every layer before the decoder (Xu et al., 2018). Normalization
                is controlled by ``normalize_output``. Intended for link
                prediction. Defaults to ``False``.
            add_lp (bool, optional): Appends one fixed stalk channel diffusing
                with the signless Laplacian D+A (reference ``--add_lp``).
            add_hp (bool, optional): Appends one fixed stalk channel diffusing
                with the standard Laplacian D-A (reference ``--add_hp``).
            sparse_learner (bool, optional): Stalk-summed map-generator input,
                shrinking the learner from 2*final_d*hidden to 2*hidden inputs.
            second_linear (bool, optional): Adds a second encoder linear stage.
            use_edge_weights: Learned symmetric per-edge scalars rescaling
                orthogonal maps (reference edge_weights). Defaults to False.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one NSP layer")

        self.stalk_dim = stalk_dim
        # Fixed lp/hp channels extend the stalk beyond the learned dims.
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
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

        extra_kwargs = variant.layer_kwargs.copy()
        if variant == NSPVariant.ORTHOGONAL:
            extra_kwargs["orth_strategy"] = orth_strategy
            extra_kwargs["use_edge_weights"] = use_edge_weights
        if variant == NSPVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
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
            assert isinstance(layer, BaseNSPConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the NSP encoder, propagation layers, and decoder.

        Args:
            x (Tensor): Raw node features with shape [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format [2, num_edges].

        Returns:
            Tensor: Node outputs with shape [num_nodes, out_channels].
        """
        x_stalk = self.encoder(self.input_dropout_layer(x))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)  # [N, final_d, f]

        # Augment with zero history stalk.
        x_stalk = torch.cat([x_stalk, torch.zeros_like(x_stalk)], dim=1)  # [N, 2d, f]

        layer_reps = []
        for layer in self.layers:
            x_curr = x_stalk[:, : self.final_d, :]
            x_feat = self.dropout_layer(x_curr.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index)
            if self.jknet:
                h = x_stalk[:, : self.final_d, :].reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk[:, : self.final_d, :].reshape(
                x_stalk.size(0), -1
            )  # [N, d*f]

        if self.normalize_output and not self.jknet:
            # L2-normalise final representation - Lv et al. (2021).
            # Skipped when jknet=True since per-layer normalisation was already applied.
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["NSPVariant", "NSPModel"]
