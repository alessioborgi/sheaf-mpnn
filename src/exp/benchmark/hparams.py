# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Resolved best-known hyperparameters for each side of the benchmark, and
the check that keeps their architectures directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, model_validator

from exp.config import Config, DatasetConfig
from exp.sweeps.sweep import _build_cfg

_REQUIRED_BEST_PARAMS = ("variant", "stalk_dim", "hidden_dim", "num_layers")


class OursHParams(BaseModel):
    """Matches the ``{model, best_params}`` YAML shape written by
    ``exp.sweeps.sweep._save_best_config``. That format doesn't always
    include ``variant`` (only the architecture-sweep hyperparameters), but
    the benchmark needs it explicit rather than silently falling back to
    ``ModelConfig``'s default -- fail fast instead of building the wrong
    variant.
    """

    model: str
    best_params: dict[str, float | int | str | bool]

    @model_validator(mode="after")
    def _require_architecture_fields(self) -> OursHParams:
        missing = [k for k in _REQUIRED_BEST_PARAMS if k not in self.best_params]
        if missing:
            raise ValueError(f"best_params is missing required field(s): {missing}")
        return self

    @property
    def stalk_dim(self) -> int:
        return int(self.best_params["stalk_dim"])

    @property
    def hidden_dim(self) -> int:
        return int(self.best_params["hidden_dim"])

    @property
    def num_layers(self) -> int:
        return int(self.best_params["num_layers"])

    def to_config(self, dataset: str, root: str = "exp/data") -> Config:
        base_cfg = Config(dataset=DatasetConfig(name=dataset, root=root))
        return _build_cfg(
            base_cfg,
            self.model,
            cast(dict[str, float | int | str | list[int]], self.best_params),
        )


class OriginalHParams(BaseModel):
    """Flat mapping of the original repo's ``exp/parser.py`` flags this
    benchmark needs to resolve per (dataset, variant); ``dataset``, ``seed``,
    ``cuda``, ``folds``, ``model``, ``entity``, and ``evectors`` are supplied
    by the adapter/entrypoint instead of this file.
    """

    d: int
    layers: int
    hidden_channels: int
    epochs: int = 1500
    lr: float = 0.01
    weight_decay: float = 5e-4
    sheaf_decay: float | None = None
    early_stopping: int = 200
    stop_strategy: str = "loss"
    normalised: bool = True
    deg_normalised: bool = False
    linear: bool = False
    input_dropout: float = 0.0
    dropout: float = 0.0
    left_weights: bool = True
    right_weights: bool = True
    add_lp: bool = False
    add_hp: bool = False
    use_act: bool = True
    second_linear: bool = False
    orth: str = "householder"
    sheaf_act: str = "tanh"
    edge_weights: bool = True
    sparse_learner: bool = False


def load_ours_hparams(path: Path) -> OursHParams:
    return OursHParams.model_validate(yaml.safe_load(Path(path).read_text()))


def load_original_hparams(path: Path) -> OriginalHParams:
    return OriginalHParams.model_validate(yaml.safe_load(Path(path).read_text()))


def validate_matched_architecture(
    ours: OursHParams, original: OriginalHParams, *, dataset: str, variant: str
) -> None:
    mismatches = []
    file_variant = str(ours.best_params["variant"])
    if file_variant != variant:
        mismatches.append(
            f"ours best_params variant {file_variant!r} != job variant {variant!r}"
        )
    if ours.stalk_dim != original.d:
        mismatches.append(f"stalk_dim {ours.stalk_dim} != d {original.d}")
    if ours.hidden_dim != original.hidden_channels:
        mismatches.append(
            f"hidden_dim {ours.hidden_dim} !="
            f" hidden_channels {original.hidden_channels}"
        )
    if ours.num_layers != original.layers:
        mismatches.append(f"num_layers {ours.num_layers} != layers {original.layers}")
    if mismatches:
        raise ValueError(
            f"Architecture mismatch for {dataset}/{variant}: {'; '.join(mismatches)}"
        )
