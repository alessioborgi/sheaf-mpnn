# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Typed configuration dataclasses for the NSD benchmark runner.

Consumed by ``tyro.cli`` in ``exp/run.py``.  Per-dataset preset defaults live
in ``exp/presets.py`` and are injected via ``tyro.cli(Config, default=...)``,
so every field remains overridable from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Annotated, Literal, TypeVar

import tyro
from tyro.constructors import PrimitiveConstructorSpec


class ModelType(StrEnum):
    NSD = auto()
    BSNN = auto()
    DIRECTED_NSD = auto()
    NLSD = auto()
    BUNN = auto()
    NSP = auto()
    SHEAF_ATTN = auto()
    SHEAF_ADJ = auto()
    DNSD = auto()
    POLY_NSD = auto()
    CONN_NSD = auto()
    POLY_HET_NSD = auto()
    HET_NSD = auto()
    HET_NSP = auto()
    HET_SHEAF_ATTN = auto()
    ID_SHEAF = auto()
    JOINT_DIFF = auto()
    ROT_INV = auto()
    CSNN = auto()
    HET_ID_SHEAF = auto()
    HET_JOINT_DIFF = auto()
    HET_ROT_INV = auto()
    HET_DIRECTED_NSD = auto()
    HET_NLSD = auto()
    HET_CSNN = auto()
    HET_BSNN = auto()
    HET_SHEAF_ADJ = auto()
    HET_DNSD = auto()
    HET_CONN_NSD = auto()
    HET_BUNN = auto()


class GCPooling(StrEnum):
    """Graph-level pooling strategy for graph classification."""

    MEAN = auto()
    ADD = auto()
    MAX = auto()
    SAG = auto()
    SHEAF_POOL = auto()


E = TypeVar("E", bound=StrEnum)


def _case_insensitive_enum(
    enum_cls: type[E],
) -> PrimitiveConstructorSpec[E]:
    """Tyro parser for a StrEnum flag that accepts ``nsd``, ``NSD`` or ``Nsd``.

    tyro's default enum rule only accepts member names (upper-case here); the
    YAML sweep configs and the registries use the lower-case values.
    """
    choices = tuple(m.value for m in enum_cls)

    def from_str(args: list[str]) -> E:
        try:
            return enum_cls(args[0].lower())
        except ValueError:
            raise ValueError(
                f"invalid choice {args[0]!r}; expected one of {', '.join(choices)}"
                " (case-insensitive)"
            ) from None

    return PrimitiveConstructorSpec(
        nargs=1,
        metavar="{" + ",".join(choices) + "}",
        instance_from_str=from_str,
        is_instance=lambda x: isinstance(x, enum_cls),
        str_from_instance=lambda m: [m.value],
    )


class ModelVariant(StrEnum):
    DIAGONAL = auto()
    GENERAL = auto()
    ORTHOGONAL = auto()
    GENERAL_ATTENTION = auto()
    ORTHOGONAL_ATTENTION = auto()
    SHEAFAN = auto()
    RES_SHEAFAN = auto()


@dataclass
class DatasetConfig:
    """Dataset identity and storage location."""

    name: str = "cora"
    root: str = "exp/data"
    # Keep the graph directed.  Runs use the undirected variant by default
    # (the graph is symmetrised after loading); pass ``--directed`` to train on
    # the original directed edges.  No-op for datasets that are already
    # undirected.  Exposed as a top-level ``--directed`` flag for convenience.
    directed: Annotated[bool, tyro.conf.arg(name="directed", prefix_name=False)] = False


@dataclass
class ModelConfig:
    """Sheaf architecture hyperparameters."""

    type: Annotated[ModelType, _case_insensitive_enum(ModelType)] = ModelType.NSD
    variant: Literal[
        "diagonal",
        "general",
        "orthogonal",
        "low_rank",
        "general_attention",
        "orthogonal_attention",
        "sheafan",
        "res_sheafan",
        "conformal",
        "non_conformal",
    ] = "general"
    stalk_dim: int = 4
    hidden_dim: int = 16
    num_layers: int = 2
    alpha: float = 1.0
    # False freezes alpha at its init value (Bodnar-exact: no learnable step).
    learn_alpha: bool = True
    # JointDiff-specific: fixed step size of the restriction-map ODE update.
    beta: float = 0.1
    # RotInv-specific: evolve maps across layers via the previous layer's maps.
    time_dep: bool = True
    rank: int = 1
    # SheafAttn-specific
    num_heads: int = 1
    leaky_relu_slope: float = 0.2
    clamp_val: float = 10.0
    orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley"
    # Reference NSD options: fixed low/high-pass stalk channels, stalk-summed
    # sparse map learner, and a second encoder linear (Bodnar et al. flags).
    add_lp: bool = False
    add_hp: bool = False
    sparse_learner: bool = False
    second_linear: bool = False
    # Learned symmetric edge weights for orthogonal maps (reference
    # edge_weights; bundle-only in Bodnar et al.).
    edge_weights: bool = False
    # DNSD-specific (Bourgerie et al., 2026): odd swaps the inherited ELU
    # activation for tanh; gate applies a learnable per-(node, stalk)
    # sigmoid gate to each layer's diffusion term. Shared by het_dnsd.
    # layer_norm (above) is reused for DNSD's per-layer post-norm; the dnsd /
    # het_dnsd presets switch it on, matching the paper's always-on convention.
    odd: bool = False
    gate: bool = False
    # BSNN-specific (Gillespie et al., 2025): the sheaf is a latent random
    # variable and training maximizes the ELBO.
    # use_kl: include the KL regularizer in the training loss (the paper's
    #   reported models train with the KL term enabled).
    use_kl: bool = True
    # kl_weight: maximum weight lambda of the KL term.
    kl_weight: float = 1.0
    # kl_anneal: "cyclical" ramps the KL weight 0 -> kl_weight with a sigmoid
    #   every kl_cycle epochs (Fu et al., 2019, as in the reference BSNN);
    #   "constant" keeps it at kl_weight.
    kl_anneal: Literal["cyclical", "constant"] = "cyclical"
    # kl_cycle: epochs per cyclical annealing cycle (reference: 40).
    kl_cycle: int = 40
    # ensemble_size: stochastic forward passes averaged at evaluation time.
    ensemble_size: int = 3
    # NLSD-specific: which nonlinearity Phi sits inside the sheaf Laplacian
    # L^Phi = delta^T Phi delta (Zaghen et al., 2024).  "mlp" is the paper's
    # learned Phi; the threshold variants keep the hand-crafted opinion-dynamics
    # gate but learn its threshold; "identity" collapses NLSD back to NSD.
    nlsd_nonlinearity: Literal[
        "mlp",
        "scalar_threshold",
        "edge_threshold",
        "identity",
    ] = "mlp"
    # Potential whose derivative gates the threshold variants (ignored by "mlp").
    nlsd_psi: Literal["linear", "cos"] = "linear"
    # Depth of the learned Phi (Linear+ReLU blocks); only used by "mlp".
    nlsd_mlp_layers: int = 2
    # Initial threshold for the "scalar_threshold" variant.
    nlsd_init_threshold: float = 1.0
    # BuNN-specific (Bamberger et al., ICLR 2025): message *diffusion* on a flat
    # vector bundle.  stalk_dim doubles as the bundle dimension d (2 in the
    # paper's experiments) and bunn_num_bundles is how many such bundles run in
    # parallel, so a node state is bunn_num_bundles * stalk_dim * hidden_dim wide.
    # Shared by het_bunn.
    bunn_num_bundles: int = 4
    # How the heat kernel exp(-t L) is evaluated -- the paper's two algorithms
    # for the same layer: "taylor" is the truncated series (cheap, small t),
    # "spectral" the exact eigendecomposition (any t, dense: small/medium graphs).
    bunn_solver: Literal["taylor", "spectral"] = "taylor"
    # Diffusion time t -- the scale at which messages diffuse.  The paper's
    # grid is 0.1/1/10/100.
    bunn_tau: float = 1.0
    # Learn one diffusion time per bundle (the reference does so for "spectral").
    bunn_learn_tau: bool = False
    # Taylor truncation degree ("taylor" only); 0 derives the smallest degree
    # that is accurate for bunn_tau.  An explicit value below that threshold is
    # rejected: the truncated series diverges rather than approximating
    # exp(-tau*L), which would read as diverged training, not misconfiguration.
    bunn_max_degree: int = 0
    # Truncate the spectral solver to this many low-frequency eigenpairs (the
    # paper's "truncated spectral solution"); 0 keeps the full, exact spectrum.
    bunn_num_eigs: int = 0
    # Network phi computing the per-node bundle maps: a node-level MLP, or the
    # paper's sum-aggregation GNN (which reads the graph structure). het_bunn's
    # use_edge_types needs "sum_gnn": edge types enter as relation-specific
    # neighbour weights, and an MLP never reads the edges.
    bunn_phi: Literal["mlp", "sum_gnn"] = "mlp"
    bunn_phi_layers: int = 2
    # Share one phi across layers (paper: "shared phi") instead of one per layer.
    bunn_shared_phi: bool = False
    # Laplacian driving the diffusion ("spectral" requires "sym").
    bunn_normalize: Literal["sym", "rw"] = "sym"
    # PolyNSD-specific: polynomial filter degree and Chebyshev scaling
    degree: int = 3
    lambda_max: float = 2.0
    hp_alpha_init: float = 0.0
    # CSNN-specific: the cooperative sheaf learner is a GNN backbone rather
    # than a single linear map, so it carries its own depth/width knobs.
    # gnn_default 1 and 2 are the two reference GraphSAGE presets; 0 builds a
    # custom stack from gnn_type (and is the only path that reads edge_attr).
    gnn_layers: int = 0
    gnn_hidden: int = 16
    gnn_default: int = 1
    gnn_type: Literal["GCN", "GAT", "SAGE", "SGC", "GPS", "NNConv", "SumGNN"] = "SAGE"
    gnn_residual: bool = False
    linear_emb: bool = False
    pe_size: int = 0
    left_weights: bool = True
    right_weights: bool = True
    use_act: bool = True
    layer_norm: bool = False
    batch_norm: bool = False
    # Depth of the CSNN classification MLP (reference ``--num_heads``). Named
    # apart from ``num_heads`` above, which means attention heads for SheafAttn.
    csnn_head_layers: int = 1
    # Graph classification specific
    pooling: Annotated[GCPooling, _case_insensitive_enum(GCPooling)] = GCPooling.MEAN
    # Heterogeneous context flags (ignored for homogeneous ModelType variants)
    num_node_types: int = 0
    num_edge_types: int = 0
    use_features: bool = True
    use_node_types: bool = False
    use_edge_types: bool = False
    ensemble: bool = False
    # Output normalisation & jumping knowledge
    # normalize_output: L2-normalize the representation before the decoder
    #   (Lv et al., 2021).  If jknet is True, per-layer outputs are normalized
    #   before concatenation.
    # jknet: concatenate hidden states across all layers before the decoder
    #   (Xu et al., 2018). Normalization is controlled by normalize_output.
    #   Intended for link prediction - leave False for node/graph classification.
    normalize_output: bool = False
    jknet: bool = False


@dataclass
class RegConfig:
    """Regularisation hyperparameters."""

    input_dropout: float = 0.0
    dropout: float = 0.0


@dataclass
class OptimConfig:
    """Optimisation schedule."""

    lr: float = 0.01
    weight_decay: float = 5e-4
    # Separate weight decay for the sheaf map generators (reference
    # sheaf_decay); None falls back to weight_decay.
    sheaf_decay: float | None = None
    # Separate learning rates for the NLSD nonlinearity threshold and the
    # per-stalk diffusion gate (reference threshold_lr / alpha_lr); None falls
    # back to lr.  Ignored by models that expose neither parameter.
    threshold_lr: float | None = None
    alpha_lr: float | None = None
    epochs: int = 1000
    early_stopping: int = 200
    stop_strategy: Literal["loss", "acc"] = "loss"
    batch_size: int = 1


@dataclass
class CVConfig:
    """Cross-validation setup."""

    folds: int = 10
    seed: int = 42
    min_acc: float = 0.0


@dataclass
class HardwareConfig:
    """Hardware selection and data loading performance."""

    cuda: int = 0
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class WandBConfig:
    """Weights & Biases integration."""

    enabled: bool = False
    entity: str | None = None
    project: str | None = None


@dataclass
class Config:
    """Root configuration for a benchmark run."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
