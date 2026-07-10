# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Pydantic models for the hyperparameter sweeps."""

import dataclasses
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    model_validator,
)


class FloatParam(BaseModel):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False


class IntParam(BaseModel):
    type: Literal["int"]
    low: int
    high: int
    log: bool = False


class CategoricalParam(BaseModel):
    type: Literal["categorical"]
    choices: list[bool | str | int | float | list[int]]


class OptunaConfig(BaseModel):
    n_trials: int = 100
    n_seeds_per_trial: int = 1
    """How many independent seeds to run per trial.  Each seed re-initialises
    model weights and (when > 1) uses a different data fold, giving a mean +/- std
    estimate of the val metric for every hyperparameter configuration."""
    std_weight: float = 0.0
    """Penalty weight for variance.  Objective = mean - std_weight * std.
    Set to 0 to optimise pure mean; increase (e.g. 0.5-1.0) to prefer
    configurations that are both good *and* stable across seeds."""
    study_name: str = "nsd-sweep"
    storage: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    nruns_per_study: int = 0
    """Deprecated and ignored: sweeps always log one WandB run per trial
    (as_multirun), so each trial is its own row in the project table."""
    cuda: int = 0
    seed: int = 42


class DatasetSpec(BaseModel):
    """Dataset identity, used to override the preset's dataset from the YAML."""

    name: str
    root: str = "exp/data"


def _valid_sweep_param_names() -> frozenset[str]:
    from exp.config import ModelConfig, OptimConfig, RegConfig

    return frozenset(
        f.name
        for cls in (ModelConfig, RegConfig, OptimConfig)
        for f in dataclasses.fields(cls)
    )


class SweepConfig(BaseModel):
    model: str
    dataset: DatasetSpec | None = None
    """Override the preset's dataset. If omitted, the preset (or default) is used."""
    search_space: dict[
        str, Annotated[FloatParam | IntParam | CategoricalParam, Discriminator("type")]
    ]
    config: OptunaConfig = Field(default_factory=OptunaConfig)

    @model_validator(mode="after")
    def _validate_model_type(self) -> "SweepConfig":
        from exp.config import ModelType

        valid = {m.value for m in ModelType}
        if self.model not in valid:
            raise ValueError(
                f"Unknown model {self.model!r}. Valid values: {sorted(valid)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_search_space_keys(self) -> "SweepConfig":
        valid = _valid_sweep_param_names()
        unknown = sorted(set(self.search_space) - valid)
        if unknown:
            raise ValueError(
                f"Unknown search_space parameter(s): {unknown}. "
                f"Must be fields of ModelConfig, RegConfig, or OptimConfig."
            )
        return self
