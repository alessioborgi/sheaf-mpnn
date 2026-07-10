# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Typed configuration dataclasses for the NSD benchmark runner.

Consumed by ``tyro.cli`` in ``exp/run.py``.  Per-dataset preset defaults live
in ``exp/registries/presets.py`` and are injected via
``tyro.cli(Config, default=...)``,
so every field remains overridable from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal


class ModelType(StrEnum):
    NSD = auto()


@dataclass
class DatasetConfig:
    """Dataset identity and storage location."""

    name: str = "cora"
    root: str = "exp/data"


@dataclass
class ModelConfig:
    """NSD architecture hyperparameters."""

    type: ModelType = ModelType.NSD
    variant: Literal[
        "diagonal",
        "general",
        "orthogonal",
        "low_rank",
        "general_attention",
        "orthogonal_attention",
    ] = "general"
    stalk_dim: int = 4
    hidden_dim: int = 16
    num_layers: int = 2
    alpha: float = 1.0
    # False freezes alpha at its init value (Bodnar-exact: no learnable step).
    learn_alpha: bool = True
    rank: int = 1
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
