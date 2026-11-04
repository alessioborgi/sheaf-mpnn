# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from enum import Enum, auto
from typing import Any, Literal

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.nlsd.nlsd_layers import (
    BaseHetNLSDConv,
    DiagonalHetNLSDConv,
    GeneralHetNLSDConv,
    LowRankHetNLSDConv,
    OrthogonalHetNLSDConv,
)
from sheaf_mpnn.nlsd.nlsd_model import NLSDNonlinearity
from sheaf_mpnn.nlsd.nonlinearity import PSI_KINDS, PsiKind


class HetNLSDVariant(Enum):
    """Restriction-map family for the heterogeneous NLSD model.

    Mirrors :class:`~sheaf_mpnn.nlsd.nlsd_model.NLSDVariant`: the paper defines
    NLSD on ``DIAGONAL`` and ``ORTHOGONAL`` only, and ``GENERAL`` / ``LOW_RANK``
    are framework extensions sharing the same block ``(D+I)^{-1/2}`` coboundary
    normalisation. There are no attention variants -- NLSD has no attention
    learner, in the het setting as in the homogeneous one.
    """

    DIAGONAL = auto()
    ORTHOGONAL = auto()
    GENERAL = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self) -> type[BaseHetNLSDConv]:
        mapping: dict[HetNLSDVariant, type[BaseHetNLSDConv]] = {
            HetNLSDVariant.DIAGONAL: DiagonalHetNLSDConv,
            HetNLSDVariant.ORTHOGONAL: OrthogonalHetNLSDConv,
            HetNLSDVariant.GENERAL: GeneralHetNLSDConv,
            HetNLSDVariant.LOW_RANK: LowRankHetNLSDConv,
        }
        return mapping[self]

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        rank: int = 1,
    ) -> dict[str, Any]:
        if self is HetNLSDVariant.ORTHOGONAL:
            return {"orth_strategy": orth_strategy, "clamp_val": clamp_val}
        if self is HetNLSDVariant.LOW_RANK:
            return {"rank": rank}
        return {}


class HetNLSDModel(nn.Module):
    r"""Heterogeneous Nonlinear Sheaf Diffusion (HetNLSD) model.

    The het counterpart of :class:`~sheaf_mpnn.nlsd.nlsd_model.NLSDModel`:
    identical encoder-layers-decoder pipeline and identical diffusion operator
    :math:`L^\Phi_\mathcal{F} = \delta^\top \circ \Phi \circ \delta`, but each
    layer's restriction maps are generated from the heterogeneous context
    (node features, node types, edge types, or a per-edge-type ensemble).

    The encoder is shared across all node types (shared cochain projection),
    matching the existing heterogeneous experiment pipeline, which flattens
    ``HeteroData`` via PyG's ``to_homogeneous()`` before passing it here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetNLSDVariant = HetNLSDVariant.ORTHOGONAL,
        nonlinearity: NLSDNonlinearity = NLSDNonlinearity.MLP,
        psi: PsiKind = "linear",
        num_mlp_layers: int = 2,
        init_threshold: float = 1.0,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_self_loops: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        rank: int = 1,
        add_eps: bool = True,
        per_stalk_alpha: bool | None = None,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        learn_alpha: bool = True,
    ):
        r"""Initializes a het-NLSD model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node.
            stalk_dim: Stalk dimension. Node states are ``[stalk_dim,
                hidden_dim]``.
            hidden_dim: Feature dimension inside each stalk channel; also the
                hidden width of the het map-generator MLPs.
            num_layers: Number of het-NLSD convolution layers. Must be positive.
            variant: Restriction-map family.
            nonlinearity: Which :math:`\Phi` sits between the coboundary and
                its transpose. ``MLP`` is the paper's learned nonlinearity.
            psi: For the threshold variants, the potential whose derivative
                gates the diffusion. Ignored by ``MLP`` and ``IDENTITY``.
            num_mlp_layers: Depth of the learned :math:`\Phi`; ``MLP`` only.
            init_threshold: Initial threshold for ``SCALAR_THRESHOLD``.
            alpha: Initial diffusion step size per layer.
            num_node_types: Number of node types in the graph.
            num_edge_types: Number of edge types in the graph.
            use_features: Feed node features to the map generators.
            use_node_types: Feed one-hot node types to the map generators.
            use_edge_types: Feed one-hot edge types to the map generators.
            ensemble: Route each edge through its edge type's own generator.
            add_self_loops: If ``True``, self-loops are added before diffusing.
            orth_strategy: Orthogonal parameterisation for ``ORTHOGONAL``.
            clamp_val: Clamp on the orthogonal-transform parameters.
            rank: Rank of each restriction map for ``LOW_RANK``.
            add_eps: Learnable per-stalk rescaling of the skip term.
            per_stalk_alpha: Whether the diffusion term carries the reference
                per-stalk gate. Defaults to the choice the reference makes for
                the selected ``nonlinearity``.
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on stalk features between layers.
            normalize_output: L2-normalise the representation before decoding.
            jknet: Concatenate every layer's hidden state before the decoder.
            learn_alpha: If ``False``, ``alpha`` stays at its initial value.

        Raises:
            ValueError: On non-positive dimensions/layers or an unknown ``psi``.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetNLSD layer")
        if psi not in PSI_KINDS:
            raise ValueError(f"psi must be one of {PSI_KINDS}, got {psi!r}")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
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

        extra_kwargs = variant.build_kwargs(
            orth_strategy=orth_strategy, clamp_val=clamp_val, rank=rank
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
                    alpha=alpha,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    per_stalk_alpha=per_stalk_alpha,
                    dropout=dropout,
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
        for layer in self.layers:
            assert isinstance(layer, BaseHetNLSDConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs the encoder, het-NLSD diffusion layers, and decoder.

        Args:
            x (Tensor): Raw node features ``[N, in_channels]``.
            edge_index (LongTensor): Graph connectivity ``[2, E]``.
            node_types (LongTensor, optional): Node-type indices ``[N]``.
            edge_types (LongTensor, optional): Edge-type indices ``[E]``.

        Returns:
            Tensor: Node outputs ``[N, out_channels]``.
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

        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []
        for i, layer in enumerate(self.layers):
            x_feat = x_stalk.reshape(x_stalk.size(0), -1)
            # Reference: the layer-0 learner sees the un-dropped features.
            if i > 0:
                x_feat = self.dropout_layer(x_feat)
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


__all__ = ["HetNLSDVariant", "HetNLSDModel"]
