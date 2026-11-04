# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Pydantic models for the efficiency-benchmark manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_VARIANT_MAP: dict[str, str] = {
    "diagonal": "DiagSheaf",
    "orthogonal": "BundleSheaf",
    "general": "GeneralSheaf",
}
KNOWN_VARIANTS: frozenset[str] = frozenset(DEFAULT_VARIANT_MAP)


class RepeatsConfig(BaseModel):
    folds: int = 10
    efficiency_seeds: int = 20
    efficiency_epoch_cap: int = 30
    warmup_epochs: int = 5

    @model_validator(mode="after")
    def _validate_warmup(self) -> RepeatsConfig:
        if self.warmup_epochs >= self.efficiency_epoch_cap:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be < "
                f"efficiency_epoch_cap ({self.efficiency_epoch_cap})"
            )
        return self


class OursSideConfig(BaseModel):
    python: str = ".venv/bin/python"
    config_dir: Path


class OriginalSideConfig(BaseModel):
    python: str
    repo_path: Path
    variant_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_VARIANT_MAP)
    )
    config_dir: Path


class SidesConfig(BaseModel):
    ours: OursSideConfig
    original: OriginalSideConfig


class SlurmConfig(BaseModel):
    partition: str
    account: str
    gres: str = "gpu:1"
    cpus_per_task: int = 4
    mem_gb: int = 16
    timeout_min: int = 120
    slurm_array_parallelism: int = 20


class StatsConfig(BaseModel):
    test: Literal["wilcoxon"] = "wilcoxon"
    alpha: float = 0.05


class BenchmarkManifest(BaseModel):
    name: str
    datasets: list[str]
    variants: list[str]
    modes: list[Literal["convergence", "efficiency"]] = ["convergence", "efficiency"]
    repeats: RepeatsConfig = Field(default_factory=RepeatsConfig)
    sides: SidesConfig
    results_dir: Path
    slurm: SlurmConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)

    @model_validator(mode="after")
    def _resolve_paths(self) -> BenchmarkManifest:
        """Make every path-like field absolute.

        Jobs for the "original" side run with ``cwd`` set to the vendored
        repo, so a relative interpreter path would fail at exec time and
        relative ``--hparams``/``--out`` paths would resolve against the
        wrong directory. Resolving at load time (relative to the CWD the
        manifest was loaded from) keeps all of those stable; resolving an
        already-absolute path is a no-op.
        """
        self.sides.ours.python = str(Path(self.sides.ours.python).resolve())
        self.sides.ours.config_dir = Path(self.sides.ours.config_dir).resolve()
        self.sides.original.python = str(Path(self.sides.original.python).resolve())
        self.sides.original.repo_path = Path(self.sides.original.repo_path).resolve()
        self.sides.original.config_dir = Path(self.sides.original.config_dir).resolve()
        self.results_dir = Path(self.results_dir).resolve()
        return self

    @model_validator(mode="after")
    def _validate_variants(self) -> BenchmarkManifest:
        unknown = sorted(set(self.variants) - KNOWN_VARIANTS)
        if unknown:
            raise ValueError(
                f"Unknown variant(s) {unknown}; must be a subset of "
                f"{sorted(KNOWN_VARIANTS)}"
            )
        missing_map = sorted(set(self.variants) - set(self.sides.original.variant_map))
        if missing_map:
            raise ValueError(
                f"sides.original.variant_map is missing entries for {missing_map}"
            )
        return self


def load_manifest(path: Path) -> BenchmarkManifest:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return BenchmarkManifest.model_validate(raw)
