# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.nsd.poly_nsd_layers import (
    BasePolyNSDConv,
    DiagonalPolyNSDConv,
    GeneralPolyNSDConv,
    LowRankPolyNSDConv,
    OrthogonalPolyNSDConv,
)


class PolyNSDVariant(Enum):
    """Restriction-map family for the polynomial NSD model.

    Mirrors ``NSDVariant`` but maps to the polynomial layer classes.
    ``ORTHOGONAL_ATTENTION`` and ``GENERAL_ATTENTION`` set ``use_attention=True``
    in the layer kwargs, matching the first-order NSD convention.
    """

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self):
        """Returns the polynomial NSD conv class for this variant."""
        mapping = {
            PolyNSDVariant.DIAGONAL: DiagonalPolyNSDConv,
            PolyNSDVariant.GENERAL: GeneralPolyNSDConv,
            PolyNSDVariant.ORTHOGONAL: OrthogonalPolyNSDConv,
            PolyNSDVariant.LOW_RANK: LowRankPolyNSDConv,
            PolyNSDVariant.GENERAL_ATTENTION: GeneralPolyNSDConv,
            PolyNSDVariant.ORTHOGONAL_ATTENTION: OrthogonalPolyNSDConv,
        }
        return mapping[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments forwarded to the layer constructor."""
        if self in {PolyNSDVariant.DIAGONAL, PolyNSDVariant.LOW_RANK}:
            return {}
        if self in {
            PolyNSDVariant.GENERAL_ATTENTION,
            PolyNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            return {"use_attention": True}
        return {"use_attention": False}


class PolyNSDModel(nn.Module):
    """End-to-end Polynomial Neural Sheaf Diffusion (PolyNSD) model.

    Replaces the first-order NSD diffusion step in each layer with a degree-K
    polynomial spectral filter of the normalised sheaf Laplacian, evaluated via
    a three-term Chebyshev recurrence (see ``BasePolyNSDConv`` for details).

    The overall forward pass is identical to ``NSDModel``:
        1. Encode raw node features to stalk space  [N, d*f].
        2. Apply ``num_layers`` polynomial NSD convolution layers.
        3. Optionally aggregate with JKNet (Xu et al., 2018).
        4. Decode the (flat) stalk representation back to ``out_channels``.

    The only differences from ``NSDModel`` are the layer class (polynomial
    variants) and the three extra hyper-parameters that control the filter
    (``degree``, ``lambda_max``, ``hp_alpha_init``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: PolyNSDVariant = PolyNSDVariant.GENERAL,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        add_self_loops: bool = False,
        orth_strategy: str = "cayley",
        rank: int = 1,
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
        """Initialises a PolyNSD model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. num classes).
            stalk_dim: Stalk dimension d.  Each node is represented internally
                as a matrix with shape [d, hidden_dim].
            hidden_dim: Feature width f inside each stalk channel.  The encoded
                node state has size d * hidden_dim.
            num_layers: Number of polynomial NSD convolution layers.  Must be >= 1.
            variant: Restriction-map family (``PolyNSDVariant``).
            alpha: Inherited diffusion step size passed to each layer constructor
                (present for API compatibility; the polynomial forward does not
                use alpha for its diffusion step).
            degree: Polynomial filter degree K.  Each layer applies K+1 Chebyshev
                basis terms, requiring degree propagate calls per forward (one
                for Delta_F(z), shared with the high-pass skip correction, plus
                degree - 1 for the remaining recurrence steps).
            lambda_max: Spectral upper bound of the normalised sheaf Laplacian.
                Defaults to 2.0 (tight bound for symmetric normalisation).
            hp_alpha_init: Initial value of the per-layer learnable high-pass
                skip weight.  Zero means no HP correction at initialisation.
            add_self_loops: If True, self-loops are added before degree
                normalisation in each layer.
            orth_strategy: Orthogonality parameterisation for the ORTHOGONAL
                variant: "cayley" (Cayley transform) or "fasth" (Householder).
            rank: Low-rank restriction-map rank r for the LOW_RANK variant.
                Must be positive; ignored for other variants.
            input_dropout: Dropout probability applied to raw input features
                before the encoder.
            dropout: Dropout probability applied to stalk features between layers.
            normalize_output: If True, L2-normalise the final representation
                before the decoder (Lv et al., 2021).  When jknet is True, each
                layer's output is also normalised before concatenation.
            jknet: If True, collect hidden states from all layers and
                concatenate them before the decoder (Xu et al., 2018).
            add_hp: Appends one fixed stalk channel diffusing with the
                standard Laplacian D-A (reference ``--add_hp``).
            add_lp: Appends one fixed stalk channel diffusing with the
                signless Laplacian D+A (reference ``--add_lp``).
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
            raise ValueError("must have at least one PolyNSD layer")
        if degree < 1:
            raise ValueError("degree must be at least 1")

        self.stalk_dim = stalk_dim
        # Fixed lp/hp channels extend the stalk beyond the learned dims.
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.degree = degree
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet

        # context_dim = final_d * f: flat per-node context for the map generator.
        context_dim = self.final_d * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)

        # Encoder: raw features [N, in_channels] -> stalk space [N, final_d*f].
        self.encoder = nn.Linear(in_channels, context_dim)
        # Second encoder stage (reference second_linear).
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        # Build extra kwargs for the chosen variant.
        extra_kwargs = variant.layer_kwargs.copy()
        if variant in {
            PolyNSDVariant.ORTHOGONAL,
            PolyNSDVariant.ORTHOGONAL_ATTENTION,
        }:
            extra_kwargs["orth_strategy"] = orth_strategy
            extra_kwargs["use_edge_weights"] = use_edge_weights
        if variant == PolyNSDVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        # Stack of polynomial NSD convolution layers.
        # Each layer receives:
        #   stalk_dim   = d
        #   in_channels = f  (feature width, i.e. hidden_dim)
        #   hidden_dim  = hidden_dim  (MLP width for map generator)
        #   context_dim = d*f  (MLP input width)
        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    alpha=alpha,
                    degree=degree,
                    lambda_max=lambda_max,
                    hp_alpha_init=hp_alpha_init,
                    add_self_loops=add_self_loops,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    sparse_learner=sparse_learner,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        # JKNet expands the decoder input to num_layers * context_dim.
        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        """Resets encoder, decoder, and all poly NSD layer parameters."""
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BasePolyNSDConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the PolyNSD encoder, diffusion layers, and decoder.

        Args:
            x: Raw node features [N, in_channels].
            edge_index: Graph connectivity in COO format [2, E].

        Returns:
            torch.Tensor: Node outputs [N, out_channels].
        """
        # Encode: [N, in_channels] -> [N, d, f].
        x_stalk = self.encoder(self.input_dropout_layer(x))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)

        layer_reps = []
        for layer in self.layers:
            # Flatten stalk to [N, d*f] as the context vector for map generation.
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index)
            if self.jknet:
                # Collect this layer's flat representation for JKNet aggregation.
                h = x_stalk.reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk.reshape(x_stalk.size(0), -1)

        if self.normalize_output:
            # L2-normalise the final representation (Lv et al., 2021).
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["PolyNSDVariant", "PolyNSDModel"]
