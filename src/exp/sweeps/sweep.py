# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

r"""YAML-driven hyperparameter sweep using Optuna and the model registry.

Usage
-----
    # Via the unified CLI (recommended):
    sheaf sweep --yaml-path nsd_cora.yaml --preset cora

    # Direct module invocation:
    python -m exp.sweeps.sweep --yaml-path nsd_cora.yaml --preset cora

    # Without a preset (uses config defaults):
    sheaf sweep --yaml-path nsd_cora.yaml

    # Distributed sweep — add storage under config in the YAML:
    #   config:
    #     storage: sqlite:///sweep.db

Example YAML
------------
    model: nsd
    dataset:              # optional — overrides the preset's dataset
      name: texas
      root: exp/data
    search_space:
      variant:
        type: categorical
        choices: [diagonal, general, orthogonal]
      stalk_dim:
        type: int
        low: 2
        high: 8
      lr:
        type: float
        low: 0.0001
        high: 0.1
        log: true
    config:
      n_trials: 100
      study_name: nsd-texas
      storage: sqlite:///sweep.db   # optional, for distributed runs
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import tyro
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from rich.console import Console

from exp.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    ModelType,
    OptimConfig,
    RegConfig,
)
from exp.data import SheafDataModule
from exp.module import SheafLightningModule
from exp.registries.presets import preset_registry
from exp.sweeps.models import (
    CategoricalParam,
    FloatParam,
    IntParam,
    SweepConfig,
)
from sheaf_mpnn.utils import setup_torch

_console = Console()

_PruningCb: type | None = None
try:
    from optuna_integration import PyTorchLightningPruningCallback as _PruningCb
except ImportError:
    try:
        from optuna.integration import (  # type: ignore[no-redef]
            PyTorchLightningPruningCallback as _PruningCb,
        )
    except ImportError:
        pass

_MODEL_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(ModelConfig)
)
_REG_FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(RegConfig))
_OPTIM_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(OptimConfig)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suggest(
    trial: optuna.Trial,
    name: str,
    spec: FloatParam | IntParam | CategoricalParam,
) -> float | int | str | list[int]:
    if isinstance(spec, FloatParam):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, IntParam):
        return trial.suggest_int(name, spec.low, spec.high, log=spec.log)
    return trial.suggest_categorical(  # ty: ignore[no-matching-overload]
        name, spec.choices
    )


def _build_cfg(
    base_cfg: Config,
    model_type: str,
    params: dict[str, float | int | str | list[int]],
) -> Config:
    """Apply sampled hyperparameters to *base_cfg*, routing by field membership."""
    model_updates: dict[str, object] = {}
    reg_updates: dict[str, object] = {}
    optim_updates: dict[str, object] = {}

    for name, value in params.items():
        if name in _MODEL_FIELDS:
            model_updates[name] = value
        elif name in _REG_FIELDS:
            reg_updates[name] = value
        elif name in _OPTIM_FIELDS:
            optim_updates[name] = value
        else:
            raise ValueError(
                f"Unknown sweep parameter {name!r}: not a field of "
                "ModelConfig, RegConfig, or OptimConfig."
            )

    new_model = dataclasses.replace(
        base_cfg.model, type=ModelType(model_type), **model_updates
    )
    new_reg = dataclasses.replace(base_cfg.reg, **reg_updates)
    new_optim = dataclasses.replace(base_cfg.optim, **optim_updates)
    return dataclasses.replace(base_cfg, model=new_model, reg=new_reg, optim=new_optim)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _run_trial(
    trial: optuna.Trial,
    sweep_cfg: SweepConfig,
    base_cfg: Config,
) -> float:
    params = {
        name: _suggest(trial, name, spec)
        for name, spec in sweep_cfg.search_space.items()
    }
    cfg = _build_cfg(base_cfg, sweep_cfg.model, params)
    optuna_cfg = sweep_cfg.config

    val_metrics: list[float] = []
    for seed_offset in range(optuna_cfg.n_seeds_per_trial):
        seed = optuna_cfg.seed + seed_offset
        random.seed(seed)
        np.random.seed(seed)  # noqa: NPY002
        torch.manual_seed(seed)

        fold = seed_offset % cfg.cv.n_folds
        dm = SheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=fold)
        dm.setup()

        monitor = (
            "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{dm.info.metric}"
        )
        mode = "min" if cfg.optim.stop_strategy == "loss" else "max"
        module = SheafLightningModule(cfg, dm.info)

        callbacks: list = [
            EarlyStopping(
                monitor=monitor,
                patience=cfg.optim.early_stopping,
                mode=mode,
            )
        ]
        if _PruningCb is not None:
            callbacks.append(_PruningCb(trial, monitor=monitor))

        trainer = Trainer(
            max_epochs=cfg.optim.epochs,
            callbacks=callbacks,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[optuna_cfg.cuda] if torch.cuda.is_available() else "auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            log_every_n_steps=1,
        )
        trainer.fit(module, dm)

        val_result = trainer.validate(module, dm, verbose=False)
        val_metrics.append(float(val_result[0].get(f"val_{dm.info.metric}", 0.0)))

    mean = float(np.mean(val_metrics))
    std = float(np.std(val_metrics)) if optuna_cfg.n_seeds_per_trial > 1 else 0.0
    trial.set_user_attr("val_mean", mean)
    trial.set_user_attr("val_std", std)
    trial.set_user_attr("n_seeds", optuna_cfg.n_seeds_per_trial)

    return mean - optuna_cfg.std_weight * std


# ---------------------------------------------------------------------------
# WandB integration (optional)
# ---------------------------------------------------------------------------


def _make_wandb_callbacks(base_cfg: Config, sweep_cfg: SweepConfig) -> list:
    try:
        from optuna_integration.wandb import WeightsAndBiasesCallback
    except ImportError:
        _console.print(
            "optuna-integration[wandb] not installed; "
            "skipping WandB logging for Optuna study."
        )
        return []

    dm_meta = SheafDataModule(base_cfg.dataset.name, root=base_cfg.dataset.root, fold=0)
    dm_meta.setup()

    kwargs: dict = {
        "metric_name": f"val_{dm_meta.info.metric}",
        "wandb_kwargs": {
            "project": sweep_cfg.config.wandb_project,
            "entity": sweep_cfg.config.wandb_entity,
        },
    }
    if sweep_cfg.config.nruns_per_study:
        kwargs["nruns_per_study"] = sweep_cfg.config.nruns_per_study
    return [WeightsAndBiasesCallback(**kwargs)]


# ---------------------------------------------------------------------------
# Core logic (testable without CLI)
# ---------------------------------------------------------------------------


def sweep(
    yaml_path: Path,
    preset: str | None = None,
) -> None:
    """Run a YAML-driven Optuna hyperparameter sweep."""
    raw = yaml.safe_load(yaml_path.read_text())
    sweep_cfg = SweepConfig.model_validate(raw)
    base_cfg = preset_registry.get_or_default(preset)

    if sweep_cfg.dataset is not None:
        base_cfg = dataclasses.replace(
            base_cfg,
            dataset=DatasetConfig(
                name=sweep_cfg.dataset.name,
                root=sweep_cfg.dataset.root,
            ),
        )

    setup_torch(precision="high", seed=sweep_cfg.config.seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    study = optuna.create_study(
        direction="maximize",
        study_name=sweep_cfg.config.study_name,
        storage=sweep_cfg.config.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    wandb_callbacks = (
        _make_wandb_callbacks(base_cfg, sweep_cfg)
        if sweep_cfg.config.wandb_project
        else []
    )
    study.optimize(
        lambda trial: _run_trial(trial, sweep_cfg, base_cfg),
        n_trials=sweep_cfg.config.n_trials,
        callbacks=wandb_callbacks,
    )

    best = study.best_trial
    mean = best.user_attrs.get("val_mean", best.value)
    std = best.user_attrs.get("val_std", 0.0)
    n = best.user_attrs.get("n_seeds", sweep_cfg.config.n_seeds_per_trial)
    _console.print(f"\n[bold]Best trial #{best.number}[/bold]")
    _console.print(f"  val metric : {mean:.4f} +/- {std:.4f}  (n={n} seeds)")
    if sweep_cfg.config.std_weight > 0:
        _console.print(
            f"  objective  : {best.value:.4f}"
            f"  (= mean - {sweep_cfg.config.std_weight}*std)"
        )
    _console.print("  hyperparameters:")
    for k, v in best.params.items():
        _console.print(f"    {k}: {v}")

    _save_best_config(sweep_cfg, base_cfg, best.params)


def _save_best_config(
    sweep_cfg: SweepConfig,
    base_cfg: Config,
    best_params: dict,
) -> None:
    """Write the best hyperparameters to a YAML file for use with exp.run."""
    dataset = sweep_cfg.dataset.name if sweep_cfg.dataset else base_cfg.dataset.name
    filename = f"{dataset}_{sweep_cfg.model}_bestconf.yaml"

    output = {"model": sweep_cfg.model, "best_params": best_params}
    yaml.dump(output, Path(filename).open("w"), default_flow_style=False)
    _console.print(f"\nBest config saved to [cyan]{filename}[/cyan]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    yaml_path: Path,
    preset: str | None = None,
) -> None:
    r"""Entry point for ``python -m exp.sweeps.sweep``.

    Args:
        yaml_path: Path to a YAML file describing the model, search space, and
            Optuna config. See module docstring for the expected format.
        preset: Named preset to use as base config.
    """
    sweep(yaml_path=yaml_path, preset=preset)


if __name__ == "__main__":
    tyro.cli(main)
