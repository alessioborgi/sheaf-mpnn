# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Preset registry: per-dataset best-known configurations.

Presets are generated programmatically rather than written out one by one.
Every preset for a given dataset shares the same per-dataset hyperparameters
(stalk_dim, hidden_dim, num_layers, dropouts, lr, weight_decay, stop_strategy);
only the restriction-map variant changes. So the data lives in one small table
-- _NODE_HPARAMS (one row per dataset) -- and the variant grid is materialised
by the loop at the bottom. Preset keys follow ``<dataset>[_nsd_<variant>]``;
the bare ``<dataset>`` key is NSD with the dataset's default variant
(``_HP.default_variant``: ``general`` for most, ``orthogonal`` for
chameleon/squirrel, ``diagonal`` for film).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from exp.config import (
    Config,
    CVConfig,
    DatasetConfig,
    HardwareConfig,
    ModelConfig,
    ModelType,
    OptimConfig,
    RegConfig,
    WandBConfig,
)
from exp.registries.base import Registry


class PresetRegistry(Registry[str, Config]):
    """Registry for named experiment presets."""

    def get_or_default(self, name: str | None) -> Config:
        """Return the named preset, or a bare ``Config()`` if name is None."""
        if name is None:
            return Config()
        return self.get(name)


_VariantLiteral = Literal[
    "diagonal",
    "general",
    "orthogonal",
    "low_rank",
    "general_attention",
    "orthogonal_attention",
]
_StopStrategyLiteral = Literal["loss", "acc"]
_OrthLiteral = Literal["cayley", "fasth", "fasthpp", "householder"]


def generate_config(
    name: str,
    *,
    variant: _VariantLiteral,
    stalk_dim: int,
    hidden_dim: int,
    num_layers: int,
    input_dropout: float,
    dropout: float,
    lr: float,
    weight_decay: float,
    model_type: ModelType = ModelType.NSD,
    alpha: float = 1.0,
    stop_strategy: _StopStrategyLiteral = "loss",
    normalize_output: bool = False,
    jknet: bool = False,
    orth_strategy: _OrthLiteral = "cayley",
    add_lp: bool = False,
    add_hp: bool = False,
    sparse_learner: bool = False,
    second_linear: bool = False,
    edge_weights: bool = False,
    learn_alpha: bool = True,
    epochs: int = 1000,
    early_stopping: int = 200,
    sheaf_decay: float | None = None,
) -> Config:
    return Config(
        dataset=DatasetConfig(name=name),
        model=ModelConfig(
            type=model_type,
            variant=variant,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=alpha,
            normalize_output=normalize_output,
            jknet=jknet,
            orth_strategy=orth_strategy,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
            second_linear=second_linear,
            edge_weights=edge_weights,
            learn_alpha=learn_alpha,
        ),
        reg=RegConfig(input_dropout=input_dropout, dropout=dropout),
        optim=OptimConfig(
            lr=lr,
            weight_decay=weight_decay,
            stop_strategy=stop_strategy,
            epochs=epochs,
            early_stopping=early_stopping,
            sheaf_decay=sheaf_decay,
        ),
        cv=CVConfig(),
        hardware=HardwareConfig(),
        wandb=WandBConfig(),
    )


class _HP(NamedTuple):
    """Per-dataset hyperparameters shared across every family and variant."""

    stalk_dim: int
    hidden_dim: int
    num_layers: int
    input_dropout: float
    dropout: float
    lr: float
    weight_decay: float
    stop_strategy: _StopStrategyLiteral = "loss"
    # Variant used by the bare ``<dataset>`` alias (the per-dataset NSD default).
    default_variant: _VariantLiteral = "general"
    # Training budget; Pei et al. rows carry the Bodnar et al. script values.
    epochs: int = 1000
    early_stopping: int = 200
    sheaf_decay: float | None = None


def _config(
    name: str,
    model_type: ModelType,
    variant: _VariantLiteral,
    hp: _HP,
    *,
    jknet: bool = False,
    normalize_output: bool = False,
) -> Config:
    """Expand one (dataset, model_type, variant) triple into a Config."""
    return generate_config(
        name,
        model_type=model_type,
        variant=variant,
        stalk_dim=hp.stalk_dim,
        hidden_dim=hp.hidden_dim,
        num_layers=hp.num_layers,
        input_dropout=hp.input_dropout,
        dropout=hp.dropout,
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        stop_strategy=hp.stop_strategy,
        jknet=jknet,
        normalize_output=normalize_output,
        epochs=hp.epochs,
        early_stopping=hp.early_stopping,
        sheaf_decay=hp.sheaf_decay,
    )


# Homophilic + heterophilic node-classification datasets that receive the full
# family x variant grid of presets.
_NODE_HPARAMS: dict[str, _HP] = {
    "cora": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "citeseer": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "pubmed": _HP(4, 32, 2, 0.5, 0.0, 0.01, 5e-4, epochs=1500),
    "chameleon": _HP(
        4,
        32,
        5,
        0.7,
        0.0,
        0.01,
        0.0002969905682317406,
        "acc",
        default_variant="diagonal",
        epochs=1000,
        early_stopping=100,
        sheaf_decay=0.0012638885974822734,
    ),
    "squirrel": _HP(
        3,
        32,
        5,
        0.7,
        0.0,
        0.01,
        0.00011215791366362148,
        "acc",
        default_variant="orthogonal",
        epochs=1000,
        early_stopping=100,
    ),
    "chameleon_filtered": _HP(
        4, 32, 3, 0.0, 0.0, 0.01, 1e-3, default_variant="orthogonal"
    ),
    "squirrel_filtered": _HP(
        4, 32, 3, 0.0, 0.0, 0.01, 1e-3, default_variant="orthogonal"
    ),
    "cornell": _HP(
        4,
        16,
        2,
        0.2,
        0.7,
        0.02,
        0.0006914841722570725,
        default_variant="diagonal",
        epochs=500,
        sheaf_decay=0.00031764232712732976,
    ),
    "texas": _HP(
        3,
        20,
        4,
        0.0,
        0.7,
        0.02,
        5e-3,
        default_variant="orthogonal",
        epochs=1500,
    ),
    "wisconsin": _HP(
        3,
        32,
        2,
        0.0,
        0.7276458263736642,
        0.02,
        0.0006685729356079199,
        default_variant="orthogonal",
        epochs=500,
    ),
    "film": _HP(
        4, 32, 3, 0.5, 0.5, 0.005, 5e-4, default_variant="diagonal", epochs=1500
    ),
    "amazon_ratings": _HP(4, 32, 4, 0.2, 0.2, 0.01, 5e-4),
    "minesweeper": _HP(3, 16, 4, 0.0, 0.0, 0.01, 5e-4),
    "questions": _HP(4, 32, 3, 0.2, 0.2, 0.005, 5e-4),
    "roman_empire": _HP(4, 32, 4, 0.5, 0.5, 0.01, 5e-4),
    "tolokers": _HP(3, 16, 3, 0.2, 0.0, 0.01, 5e-4),
}

# Restriction-map variants: one preset per node dataset per variant, keyed
# ``<dataset>_nsd_<variant>``.
_NSD_VARIANTS: list[_VariantLiteral] = [
    "diagonal",
    "general",
    "orthogonal",
    "general_attention",
    "orthogonal_attention",
    "low_rank",
]

_PRESETS: dict[str, Config] = {}

# Node-classification datasets: bare alias (NSD, per-dataset default_variant)
# plus the full variant grid.
for _ds, _hp in _NODE_HPARAMS.items():
    _PRESETS[_ds] = _config(_ds, ModelType.NSD, _hp.default_variant, _hp)
    for _variant in _NSD_VARIANTS:
        _PRESETS[f"{_ds}_nsd_{_variant}"] = _config(_ds, ModelType.NSD, _variant, _hp)

# Bodnar et al. defaults: the plain <dataset> presets for the five scripted
# Pei et al. datasets ARE his exact configurations (BundleSheaf -> orthogonal
# + householder + edge weights; DiagSheaf -> diagonal; learn_alpha=False is
# his exact update), overriding the grid-generated aliases above.
_PRESETS["texas"] = generate_config(
    "texas",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=20,
    num_layers=4,
    input_dropout=0.0,
    dropout=0.7,
    lr=0.02,
    weight_decay=5e-3,
    orth_strategy="householder",
    edge_weights=True,
    sparse_learner=True,
    learn_alpha=False,
    epochs=1500,
)
_PRESETS["wisconsin"] = generate_config(
    "wisconsin",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=32,
    num_layers=2,
    input_dropout=0.0,
    dropout=0.7276458263736642,
    lr=0.02,
    weight_decay=0.0006685729356079199,
    orth_strategy="householder",
    edge_weights=True,
    add_lp=True,
    add_hp=True,
    learn_alpha=False,
    epochs=500,
)
_PRESETS["cornell"] = generate_config(
    "cornell",
    variant="diagonal",
    stalk_dim=4,
    hidden_dim=16,
    num_layers=2,
    input_dropout=0.2,
    dropout=0.7,
    lr=0.02,
    weight_decay=0.0006914841722570725,
    sheaf_decay=0.00031764232712732976,
    add_lp=True,
    learn_alpha=False,
    epochs=500,
)
_PRESETS["chameleon"] = generate_config(
    "chameleon",
    variant="diagonal",
    stalk_dim=4,
    hidden_dim=32,
    num_layers=5,
    input_dropout=0.7,
    dropout=0.0,
    lr=0.01,
    weight_decay=0.0002969905682317406,
    sheaf_decay=0.0012638885974822734,
    add_lp=True,
    second_linear=True,
    learn_alpha=False,
    epochs=1000,
    early_stopping=100,
    stop_strategy="acc",
)
_PRESETS["squirrel"] = generate_config(
    "squirrel",
    variant="orthogonal",
    stalk_dim=3,
    hidden_dim=32,
    num_layers=5,
    input_dropout=0.7,
    dropout=0.0,
    lr=0.01,
    weight_decay=0.00011215791366362148,
    orth_strategy="householder",
    edge_weights=True,
    add_lp=True,
    second_linear=True,
    learn_alpha=False,
    epochs=1000,
    early_stopping=100,
    stop_strategy="acc",
)

# ---------------------------------------------------------------------------
# Registry instance - populated from _PRESETS at import time
# ---------------------------------------------------------------------------

preset_registry: PresetRegistry = PresetRegistry()
for _name, _cfg in _PRESETS.items():
    preset_registry.register(_name, _cfg)
