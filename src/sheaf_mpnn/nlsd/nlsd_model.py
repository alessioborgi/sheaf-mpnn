# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.nlsd.nlsd_layers import (
    BaseNLSDConv,
    DiagonalNLSDConv,
    GeneralNLSDConv,
    LowRankNLSDConv,
    OrthogonalNLSDConv,
)
from sheaf_mpnn.nlsd.nonlinearity import (
    PSI_KINDS,
    BaseEdgeNonlinearity,
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    PsiKind,
    ScalarThresholdNonlinearity,
)


class NLSDVariant(Enum):
    """Restriction-map family.

    The paper defines NLSD on the ``diagonal`` and ``orthogonal`` families only
    (Section 4.2). ``GENERAL`` and ``LOW_RANK`` are framework extensions: the
    construction never depends on the shape of the maps, so they follow from
    the same block ``(D+I)^{-1/2}`` coboundary normalisation the linear
    families already use. Only orthogonal maps preserve the norm of the
    disagreement the threshold nonlinearities gate on -- see
    :class:`~sheaf_mpnn.nlsd.nlsd_layers.GeneralNLSDConv`.
    """

    DIAGONAL = auto()
    ORTHOGONAL = auto()
    GENERAL = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self) -> type[BaseNLSDConv]:
        mapping: dict[NLSDVariant, type[BaseNLSDConv]] = {
            NLSDVariant.DIAGONAL: DiagonalNLSDConv,
            NLSDVariant.ORTHOGONAL: OrthogonalNLSDConv,
            NLSDVariant.GENERAL: GeneralNLSDConv,
            NLSDVariant.LOW_RANK: LowRankNLSDConv,
        }
        return mapping[self]

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        use_edge_weights: bool = False,
        rank: int = 1,
    ) -> dict[str, Any]:
        if self is NLSDVariant.ORTHOGONAL:
            return {
                "orth_strategy": orth_strategy,
                "clamp_val": clamp_val,
                "use_edge_weights": use_edge_weights,
            }
        if self is NLSDVariant.LOW_RANK:
            return {"rank": rank}
        return {}


class NLSDNonlinearity(Enum):
    r"""Which edge-space map :math:`\Phi` the diffusion uses.

    ``MLP`` is the model of the paper -- :math:`\Phi` learned end to end. The
    two ``*_THRESHOLD`` members keep the hand-crafted opinion-dynamics gate of
    Hansen & Ghrist (2021) but learn its threshold, and ``IDENTITY`` collapses
    NLSD back to linear NSD (useful as a control and for parity tests).
    """

    MLP = auto()
    SCALAR_THRESHOLD = auto()
    EDGE_THRESHOLD = auto()
    IDENTITY = auto()

    @property
    def uses_per_stalk_alpha(self) -> bool:
        """Whether the reference update gates the diffusion term per stalk.

        The threshold variants learn the reference ``alphas``; the MLP variant
        subtracts the diffusion term outright.
        """
        return self in {
            NLSDNonlinearity.SCALAR_THRESHOLD,
            NLSDNonlinearity.EDGE_THRESHOLD,
        }

    def build(
        self,
        stalk_dim: int,
        hidden_dim: int,
        context_dim: int,
        psi: PsiKind = "linear",
        num_mlp_layers: int = 2,
        init_threshold: float = 1.0,
    ) -> BaseEdgeNonlinearity:
        """Instantiates a fresh nonlinearity module for one layer."""
        if self is NLSDNonlinearity.MLP:
            return MLPNonlinearity(stalk_dim, hidden_dim, num_layers=num_mlp_layers)
        if self is NLSDNonlinearity.SCALAR_THRESHOLD:
            return ScalarThresholdNonlinearity(psi=psi, init_threshold=init_threshold)
        if self is NLSDNonlinearity.EDGE_THRESHOLD:
            return EdgeThresholdNonlinearity(context_dim, psi=psi)
        return IdentityNonlinearity()


class NLSDModel(nn.Module):
    r"""End-to-end Nonlinear Sheaf Diffusion (NLSD) model.

    NLSD (Zaghen et al., GRaM @ ICML 2024) generalises NSD by inserting a
    learned nonlinearity :math:`\Phi` into the sheaf Laplacian,

    .. math::

        L^\Phi_\mathcal{F}(x)_v = \sum_{u, v \lhd e} F_{v \lhd e}^\top\,
            \phi_e\!\left(F_{v \lhd e} x_v - F_{u \lhd e} x_u\right),

    so message passing can express dynamics a linear operator cannot -- bounded
    confidence, antagonism, and anything else the data calls for.  With
    :math:`\Phi = \mathrm{id}` the model is exactly NSD.

    The wrapper keeps the family's shared skeleton: lift raw node features into
    stalk space, apply a stack of NLSD convolutions, decode the flattened stalk
    representation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: NLSDVariant = NLSDVariant.ORTHOGONAL,
        nonlinearity: NLSDNonlinearity = NLSDNonlinearity.MLP,
        psi: PsiKind = "linear",
        num_mlp_layers: int = 2,
        init_threshold: float = 1.0,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        rank: int = 1,
        add_eps: bool = True,
        per_stalk_alpha: bool | None = None,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
        sparse_learner: bool = False,
        second_linear: bool = False,
        use_edge_weights: bool = False,
        learn_alpha: bool = True,
    ):
        r"""Initializes an NLSD model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. num classes).
            stalk_dim: Stalk dimension. Each node is represented internally as a
                matrix with shape ``[stalk_dim, hidden_dim]``.
            hidden_dim: Feature dimension inside each stalk channel. The encoded
                node state has size ``stalk_dim * hidden_dim``.
            num_layers: Number of NLSD convolution layers. Must be positive.
            variant: Restriction-map family. The paper defines ``DIAGONAL`` and
                ``ORTHOGONAL``; ``GENERAL`` (``O(d^2)`` parameters per edge) and
                ``LOW_RANK`` (``2*d*rank``) are framework extensions.
            nonlinearity: Which :math:`\Phi` sits between the coboundary and
                its transpose. ``MLP`` is the paper's learned nonlinearity.
            psi: For the threshold variants, the potential whose derivative
                gates the diffusion: ``"linear"`` (bounded confidence) or
                ``"cos"``. Ignored by the ``MLP`` and ``IDENTITY`` variants.
            num_mlp_layers: Depth of the learned :math:`\Phi` (Linear+ReLU
                blocks). Only used by the ``MLP`` variant.
            init_threshold: Initial threshold for ``SCALAR_THRESHOLD``.
            alpha: Initial diffusion step size per layer. Defaults to 1.0.
            add_self_loops: If ``True``, self-loops are added before diffusing.
                Defaults to ``False``: the normalisation is already
                (deg+1)-augmented, so self-loops would augment twice.
            orth_strategy: Orthogonal parameterisation for the ``ORTHOGONAL``
                variant: "cayley", "fasth", "fasthpp", or "householder".
            clamp_val: Clamp on the orthogonal-transform parameters.
            rank: Rank of each restriction map for the ``LOW_RANK`` variant.
            add_eps: If ``True``, each layer applies the learnable per-stalk
                rescaling ``eps`` to the skip term (Eq. 7/10 of the paper).
            per_stalk_alpha: Whether the diffusion term carries the reference
                per-stalk ``(1 + tanh(alpha_gate))`` gate. Defaults to the
                choice the reference makes for the selected ``nonlinearity``
                (on for the threshold variants, off for the MLP).
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on stalk features between layers.
            normalize_output: If ``True``, L2-normalise the representation
                before the decoder (Lv et al., 2021).
            jknet: If ``True``, concatenate every layer's hidden state before
                the decoder (Xu et al., 2018). Intended for link prediction.
            sparse_learner: Stalk-summed map-generator input (reference
                LocalConcatSheafLearnerVariant).
            second_linear: Adds a second encoder linear after the ELU+dropout
                stage (reference ``--second_linear``).
            use_edge_weights: Learned symmetric per-edge scalars rescaling the
                orthogonal maps. Orthogonal variant only.
            learn_alpha: If ``False``, ``alpha`` stays fixed at its initial
                value instead of being learned.

        Raises:
            ValueError: On non-positive dimensions/layers or an unknown ``psi``.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one NLSD layer")
        if psi not in PSI_KINDS:
            raise ValueError(f"psi must be one of {PSI_KINDS}, got {psi!r}")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.variant = variant
        self.nonlinearity = nonlinearity
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim
        layer_class = variant.layer_class

        if per_stalk_alpha is None:
            per_stalk_alpha = nonlinearity.uses_per_stalk_alpha

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        extra_kwargs = variant.build_kwargs(
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
            use_edge_weights=use_edge_weights,
            rank=rank,
        )

        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    # One Phi per layer: the nonlinearity is learned per step of
                    # the diffusion, exactly like the restriction maps.
                    nonlinearity=nonlinearity.build(
                        stalk_dim=stalk_dim,
                        hidden_dim=hidden_dim,
                        context_dim=context_dim,
                        psi=psi,
                        num_mlp_layers=num_mlp_layers,
                        init_threshold=init_threshold,
                    ),
                    context_dim=context_dim,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    per_stalk_alpha=per_stalk_alpha,
                    dropout=dropout,
                    sparse_learner=sparse_learner,
                    **extra_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        if not learn_alpha:
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
            assert isinstance(layer, BaseNLSDConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        """Runs the NLSD encoder, diffusion layers, and decoder.

        Args:
            x (torch.Tensor): Raw node features ``[num_nodes, in_channels]``.
            edge_index (torch.Tensor): Graph connectivity ``[2, num_edges]``.

        Returns:
            torch.Tensor: Node outputs ``[num_nodes, out_channels]``.
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

        # Lift raw features to stalk space: [N, in_channels] -> [N, d, f].
        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.stalk_dim, self.hidden_dim)

        layer_reps = []
        for i, layer in enumerate(self.layers):
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
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


__all__ = ["NLSDVariant", "NLSDNonlinearity", "NLSDModel"]
