# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Complex encoder/convolution/readout for directed hypergraph sheaves."""

from enum import Enum, auto
from typing import Any, Literal

import torch
from torch import nn

from sheaf_mpnn.directed_hypergraph_sheaf_conv import (
    BaseDirectedHypergraphSheafConv,
    complex_activation,
    complex_dropout,
)
from sheaf_mpnn.nsd.dhsnn_layers import (
    DiagonalDHSNNConv,
    DirectedMapLearner,
    GeneralDHSNNConv,
    LowRankDHSNNConv,
    OrthogonalDHSNNConv,
)


class DHSNNVariant(Enum):
    """Restriction-map families for DHSNN; independent of graph NSD variants."""

    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    LOW_RANK = auto()

    @property
    def layer_class(self) -> type[BaseDirectedHypergraphSheafConv]:
        return {
            DHSNNVariant.DIAGONAL: DiagonalDHSNNConv,
            DHSNNVariant.GENERAL: GeneralDHSNNConv,
            DHSNNVariant.ORTHOGONAL: OrthogonalDHSNNConv,
            DHSNNVariant.LOW_RANK: LowRankDHSNNConv,
        }[self]

    @property
    def layer_kwargs(self) -> dict[str, Any]:
        return {}

    def build_kwargs(
        self,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
    ) -> dict[str, Any]:
        """Select only the arguments relevant to this restriction-map family."""
        kwargs = self.layer_kwargs
        if self == DHSNNVariant.ORTHOGONAL:
            kwargs["orth_strategy"] = orth_strategy
        if self == DHSNNVariant.LOW_RANK:
            kwargs["rank"] = rank
        return kwargs


class DHSNNModel(nn.Module):
    """Lift real input to equal real/imaginary features and retain complex states.

    A static sheaf is learned once per forward pass; dynamic sheaves learn
    fresh maps at each layer. The final real classifier sees both components.
    q=0 gives the undirected operator, but the activation, learner widths and
    classifier differ from HSNN. Supported DSHN configurations are checked
    against the upstream implementation; experimental defaults remain local.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: DHSNNVariant = DHSNNVariant.DIAGONAL,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        rank: int = 1,
        q: float = 0.25,
        activation: Literal["complex_relu", "componentwise_elu"] = "complex_relu",
        dynamic_sheaf: bool = False,
        light: bool = False,
        init_hedge: Literal["avg", "rand"] = "avg",
        orthogonal_map: Literal["native", "matrix_exp"] = "native",
        map_activation: str = "sigmoid",
        prediction: str = "MLP_var1",
        norm_type: str = "sym_block_norm",
        add_identity: bool = True,
        left_projection: bool = False,
        input_norm: bool = True,
        affine_norm: bool = True,
        classifier_hidden: int = 64,
        classifier_layers: int = 2,
        classifier_norm: str = "ln",
        sheaf_dropout: bool = False,
        special_head: bool = False,
    ) -> None:
        super().__init__()
        if (
            min(
                in_channels,
                out_channels,
                stalk_dim,
                hidden_dim,
                num_layers,
                classifier_hidden,
                classifier_layers,
            )
            <= 0
        ):
            raise ValueError(
                "channel widths, stalk_dim and num_layers must be positive"
            )
        if not isinstance(variant, DHSNNVariant):
            raise ValueError("variant must be an DHSNNVariant")
        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        context_dim = stalk_dim * hidden_dim
        self.q = q
        self.dynamic_sheaf = dynamic_sheaf
        if orthogonal_map not in ("native", "matrix_exp"):
            raise ValueError("orthogonal_map must be native or matrix_exp")
        if init_hedge not in ("avg", "rand"):
            raise ValueError("init_hedge must be avg or rand")
        self.init_hedge = init_hedge
        self.register_buffer("random_edge_features", None, persistent=False)
        self.map_learners = nn.ModuleList(
            [
                DirectedMapLearner(
                    hidden_dim,
                    stalk_dim,
                    variant.name.lower(),
                    map_activation,
                    prediction,
                    orth_strategy if orthogonal_map == "native" else orthogonal_map,
                    rank,
                    dropout if sheaf_dropout else 0.0,
                    special_head,
                )
                for _ in range(num_layers if dynamic_sheaf else 1)
            ]
        )
        self.encoder = nn.Linear(in_channels, context_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                variant.layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=2 * context_dim,
                    q=q,
                    activation=activation,
                    dropout=0.0,
                    residual=residual,
                    degree_regularization=degree_regularization,
                    norm_type=(
                        "sym_degree_norm"
                        if norm_type == "sym_block_norm"
                        else "degree_norm"
                        if norm_type == "block_norm"
                        else norm_type
                    )
                    if variant == DHSNNVariant.ORTHOGONAL
                    else norm_type,
                    add_identity=True
                    if variant == DHSNNVariant.ORTHOGONAL
                    else add_identity,
                    left_projection=left_projection,
                    input_norm=input_norm,
                    affine_norm=affine_norm,
                    light=light,
                    **variant.build_kwargs(orth_strategy=orth_strategy, rank=rank),
                )
                for _ in range(num_layers)
            ]
        )
        for layer in self.layers:
            del (
                layer.map_generator
            )  # Predictors belong to the model, once per static sheaf.
        if classifier_norm not in ("ln", "bn", "none"):
            raise ValueError("unknown classifier normalization")
        decoder = []
        width = 2 * context_dim
        for _ in range(classifier_layers - 1):
            decoder.extend([nn.Linear(width, classifier_hidden), nn.ReLU()])
            if classifier_norm != "none":
                decoder.append(
                    nn.LayerNorm(classifier_hidden)
                    if classifier_norm == "ln"
                    else nn.BatchNorm1d(classifier_hidden)
                )
            decoder.append(nn.Dropout(dropout))
            width = classifier_hidden
        decoder.append(nn.Linear(width, out_channels))
        self.decoder = nn.Sequential(*decoder)

    def reset_parameters(self) -> None:
        """Reset encoder, decoder, transforms and every restriction-map learner."""
        self.encoder.reset_parameters()
        self.random_edge_features = None
        for learner in self.map_learners:
            assert isinstance(learner, DirectedMapLearner)
            learner.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseDirectedHypergraphSheafConv)
            layer.reset_parameters()
        for module in self.decoder:
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.BatchNorm1d)):
                module.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        incidence_roles: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
        task: str = "node",
    ) -> torch.Tensor:
        """Compute raw node logits for a single directed hypergraph.

        Args:
            x: Real features [N > 0, in_channels], matching model device/dtype.
            hyperedge_index: Long indices [2, K], node IDs followed by hyperedge
                IDs in separate numbering spaces. Duplicate incidences are invalid.
            incidence_roles: Long [K] tail=-1/head=+1 labels.
            task: Prediction domain, node or hyperedge; hyperedges use sum readout.
            num_hyperedges: Optional M, including trailing empty hyperedges. If
                omitted, inferred from incidences (zero for an empty index).

        Returns:
            Node logits [N, out_channels], without a softmax.
        """
        if task not in ("node", "hyperedge"):
            raise ValueError("task must be node or hyperedge")
        if x.ndim != 2 or x.size(0) == 0 or x.size(1) != self.encoder.in_features:
            raise ValueError("x must have shape [N > 0, in_channels]")
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("x must have dtype float32 or float64")
        if (
            x.dtype != self.encoder.weight.dtype
            or x.device != self.encoder.weight.device
        ):
            raise ValueError("x must match the model's device and dtype")
        dropped = self.input_dropout_layer(x)
        # The imaginary input is detached before the shared projection, as in
        # the reference model. Encoder weights still receive both gradients.
        x_stalk = torch.complex(self.encoder(dropped), self.encoder(dropped.detach()))
        x_stalk = x_stalk.reshape(x.size(0), self.stalk_dim, self.hidden_dim)
        num_hyperedges = BaseDirectedHypergraphSheafConv._validate_incidence(
            hyperedge_index, x.size(0), num_hyperedges, x.device
        )
        from torch_geometric.utils import scatter

        edge_context = torch.complex(
            scatter(
                x_stalk.real[hyperedge_index[0]],
                hyperedge_index[1],
                dim_size=num_hyperedges,
                reduce="mean",
            ),
            scatter(
                x_stalk.imag[hyperedge_index[0]],
                hyperedge_index[1],
                dim_size=num_hyperedges,
                reduce="mean",
            ),
        )
        if self.init_hedge == "rand":
            shape = (num_hyperedges, self.encoder.in_features)
            if self.random_edge_features is None:
                self.random_edge_features = torch.randn(
                    shape, device=x.device, dtype=x_stalk.dtype
                )
            elif self.random_edge_features.shape != shape:
                raise ValueError(
                    "Random hyperedge context belongs to another graph; "
                    "reset parameters first"
                )
            edge_context = torch.complex(
                self.encoder(self.random_edge_features.real),
                self.encoder(self.random_edge_features.imag),
            ).reshape(num_hyperedges, self.stalk_dim, self.hidden_dim)
        maps = None
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, BaseDirectedHypergraphSheafConv)
            if i == 0 or self.dynamic_sheaf:
                maps = self.map_learners[i if self.dynamic_sheaf else 0](
                    x_stalk, edge_context, hyperedge_index, num_hyperedges
                )
            flat = x_stalk.reshape(x.size(0), -1)
            x_feat = torch.cat((flat.real, flat.imag), dim=-1)
            x_stalk = layer(
                x_feat,
                x_stalk,
                hyperedge_index,
                incidence_roles,
                num_hyperedges=num_hyperedges,
                activate=False,
                restriction_maps=maps,
            )
            if i < self.num_layers - 1:
                x_stalk = complex_activation(x_stalk, layer.activation)
                x_stalk = complex_dropout(x_stalk, self.dropout_layer.p, self.training)
        flat = x_stalk.reshape(x.size(0), -1)
        embedding = torch.cat((flat.real, flat.imag), dim=-1)
        if task == "hyperedge":
            embedding = scatter(
                embedding[hyperedge_index[0]],
                hyperedge_index[1],
                dim=0,
                dim_size=num_hyperedges,
                reduce="sum",
            )
        return self.decoder(embedding)


__all__ = ["DHSNNVariant", "DHSNNModel"]
