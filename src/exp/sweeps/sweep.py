# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""YAML-driven hyperparameter sweep using Optuna and the model registry.

Usage::

    # Via the unified CLI (recommended):
    sheaf sweep --yaml-path nsd_cora.yaml --preset cora

    # Direct module invocation:
    python -m exp.sweeps.sweep --yaml-path nsd_cora.yaml --preset cora

    # Without a preset (uses config defaults):
    sheaf sweep --yaml-path nsd_cora.yaml

    # Distributed sweep  add storage under config in the YAML:
    #   config:
    #     storage: sqlite:///sweep.db

Example YAML::

    model: nsd
    dataset:              # optional  overrides the preset's dataset
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
import logging
import random
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import optuna
import torch
import tyro
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState
from rich.console import Console
from rich.table import Table

from exp.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    ModelType,
    OptimConfig,
    RegConfig,
)
from exp.data import SheafDataModule
from exp.module import select_module_cls as _select_homogeneous_module_cls
from exp.registries.presets import preset_registry
from exp.resource import resource_metrics, start_resource_tracking
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
    from optuna_integration import (
        PyTorchLightningPruningCallback as _PruningCb,
    )
except ImportError:  # pragma: no cover
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


def _silence_training_noise() -> None:
    """Per-fold restore/GPU banners and full-batch dataloader hints drown the
    sweep console; drop Lightning INFO logs and the known-benign warnings.
    """
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    # Full-batch graph datasets: one Data object per epoch, workers are useless.
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    # Optuna pruning reports once per epoch; validate/test re-report the step.
    warnings.filterwarnings(
        "ignore", message="The reported value is ignored because this"
    )


# ---------------------------------------------------------------------------
# Live progress reporting
# ---------------------------------------------------------------------------

# How often (in finished trials) the full study summary is re-rendered.
_SUMMARY_EVERY = 10
_SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Variant names that shipped sweep YAMLs use as their filename stem.
_KNOWN_VARIANTS = frozenset(
    {
        "diagonal",
        "general",
        "orthogonal",
        "low_rank",
        "general_attention",
        "orthogonal_attention",
        "sheafan",
        "res_sheafan",
    }
)


def _is_het_dataset(name: str) -> bool:
    """Whether ``name`` is one of the HGB heterogeneous datasets."""
    from exp.datasets.hgb import het_dataset_registry

    return name in het_dataset_registry


def _data_module_cls(cfg: Config) -> type[SheafDataModule]:
    """DataModule class for the configured dataset.

    HGB datasets need ``HetSheafDataModule``, which loads the heterogeneous
    graph, flattens it with ``to_homogeneous()`` and keeps the node/edge type
    tensors the het families read. It subclasses ``SheafDataModule`` with the
    same constructor, so the call sites are unchanged.
    """
    if _is_het_dataset(cfg.dataset.name):
        from exp.datasets.hgb import HetSheafDataModule

        return HetSheafDataModule
    return SheafDataModule


def select_module_cls(cfg: Config):
    """Lightning module class for the configured (model, dataset) pair.

    Extends the homogeneous selector with the het modules: a ``het_*`` model on
    an HGB dataset needs ``HetSheafLightningModule``, which forwards the
    node/edge type tensors and back-fills ``num_node_types`` /
    ``num_edge_types`` from the dataset before the model is built. Homogeneous
    sweeps are routed exactly as before.
    """
    if "HET_" in str(cfg.model.type).upper() and _is_het_dataset(cfg.dataset.name):
        from exp.module import HetBayesSheafLightningModule, HetSheafLightningModule

        if cfg.model.type == ModelType.HET_BSNN:
            return HetBayesSheafLightningModule
        return HetSheafLightningModule
    return _select_homogeneous_module_cls(cfg)


def _fmt_duration(seconds: float) -> str:
    """Compact human-readable duration: 90 -> '1m30s', 4000 -> '1h06m'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def _fmt_param(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _sparkline(values: list[float], width: int = 60) -> str:
    """Render the last *width* values as an 8-level block-character strip."""
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    if hi == lo:
        return _SPARK_CHARS[0] * len(tail)
    scale = (len(_SPARK_CHARS) - 1) / (hi - lo)
    return "".join(_SPARK_CHARS[int((v - lo) * scale)] for v in tail)


class _SweepProgress:
    """Optuna callback: one status line per finished trial, plus a study
    summary (history sparkline, top trials, importances) every few trials.
    """

    _DONE = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    _COLORS = {"COMPLETE": "green", "PRUNED": "yellow", "FAIL": "red"}

    def __init__(self, n_trials: int) -> None:
        self._n_trials = n_trials

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        done = [t for t in study.trials if t.state in self._DONE]
        counts = Counter(t.state for t in done)
        durations = [t.duration.total_seconds() for t in done if t.duration]
        eta = "?"
        if durations:
            remaining = max(self._n_trials - len(done), 0)
            eta = _fmt_duration(float(np.mean(durations)) * remaining)

        color = self._COLORS.get(trial.state.name, "white")
        value = f"{trial.value:.4f}" if trial.value is not None else "-"
        try:
            best = study.best_trial
            best_txt = f"best {best.value:.4f} (#{best.number})"
        except ValueError:
            best_txt = "best -"
        _console.print(
            f"[{color}]trial {trial.number:>3} {trial.state.name}[/{color}]"
            f" obj={value} | {len(done)}/{self._n_trials} done"
            f" (C:{counts[TrialState.COMPLETE]} P:{counts[TrialState.PRUNED]}"
            f" F:{counts[TrialState.FAIL]}) | {best_txt} | eta {eta}"
        )
        if done and (len(done) % _SUMMARY_EVERY == 0 or len(done) >= self._n_trials):
            self.print_summary(study)

    def print_summary(self, study: optuna.Study) -> None:
        complete = [t for t in study.trials if t.state is TrialState.COMPLETE]
        values = [t.value for t in complete if t.value is not None]
        if not values:
            return
        _console.print(
            f"\n[bold]objective history[/bold] ({len(values)} complete trials,"
            f" min {min(values):.4f}, max {max(values):.4f})\n"
            f"  {_sparkline(values)}"
        )

        table = Table(title="top trials")
        table.add_column("trial", justify="right")
        table.add_column("objective", justify="right")
        table.add_column("val mean +/- std", justify="right")
        table.add_column("duration", justify="right")
        ranked = sorted(
            complete,
            key=lambda t: t.value if t.value is not None else float("-inf"),
            reverse=True,
        )
        for t in ranked[:5]:
            mean = t.user_attrs.get("val_mean", t.value or 0.0)
            std = t.user_attrs.get("val_std", 0.0)
            dur = _fmt_duration(t.duration.total_seconds()) if t.duration else "-"
            table.add_row(
                f"#{t.number}",
                f"{t.value:.4f}" if t.value is not None else "-",
                f"{mean:.4f} +/- {std:.4f}",
                dur,
            )
        _console.print(table)

        # Importances need a few complete trials and can fail on degenerate
        # search spaces; they are a best-effort diagnostic only.
        if len(complete) >= 4:
            try:
                importances = optuna.importance.get_param_importances(study)
            except Exception:  # noqa: BLE001
                return
            _console.print("[bold]param importance[/bold]")
            for name, imp in list(importances.items())[:10]:
                bar = "#" * max(1, round(imp * 24))
                _console.print(f"  {name:<16} {bar} {imp:.2f}")
        _console.print()


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _prior_duplicate(trial: optuna.Trial) -> FrozenTrial | None:
    """Earliest finished trial with the same params as *trial*, if any.

    Storage-backed studies keep finished trials across resumes, so this also
    catches configurations already tried (or crashed) in a previous sweep run.
    """
    done = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    for prev in trial.study.get_trials(deepcopy=False, states=done):
        if prev.number != trial.number and prev.params == trial.params:
            return prev
    return None


def _set_wandb_run_config(fields: dict) -> None:
    """Stamp identity fields on the active W&B run at trial start.

    Written to BOTH config and summary: tables/groupings built on the old
    summary-based identity keep working alongside config-based ones, and the
    end-of-trial wandb.log never runs for pruned/failed/crashed trials.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.run.config.update(fields, allow_val_change=True)
        for key, value in fields.items():
            wandb.run.summary[key] = value


def _run_trial(
    trial: optuna.Trial,
    sweep_cfg: SweepConfig,
    base_cfg: Config,
) -> float:
    params = {
        name: _suggest(trial, name, spec)
        for name, spec in sweep_cfg.search_space.items()
    }
    param_txt = " ".join(f"{k}={_fmt_param(v)}" for k, v in sorted(params.items()))
    _console.print(f"[bold cyan]trial {trial.number} start[/bold cyan] {param_txt}")

    cfg = _build_cfg(base_cfg, sweep_cfg.model, params)
    optuna_cfg = sweep_cfg.config
    _set_wandb_run_config(
        {
            "model": sweep_cfg.model,
            "dataset": cfg.dataset.name,
            "transport_type": cfg.model.variant,
            "trial_number": trial.number,
        }
    )

    # Never re-train a configuration this study already finished (or crashed
    # on) in an earlier run; prune it so the sampler moves on.
    dup = _prior_duplicate(trial)
    if dup is not None:
        dup_val = f"{dup.value:.4f}" if dup.value is not None else "-"
        _console.print(
            f"[yellow]trial {trial.number} skip[/yellow] params identical to"
            f" trial #{dup.number} ({dup.state.name}, obj={dup_val});"
            f" pruning duplicate"
        )
        _set_wandb_run_config({"pruned_duplicate_of": dup.number})
        raise optuna.TrialPruned(f"duplicate of trial {dup.number}")

    seed_metrics: dict[str, list[float]] = defaultdict(list)
    primary_key: str = ""
    for seed_offset in range(optuna_cfg.n_seeds_per_trial):
        seed = optuna_cfg.seed + seed_offset
        random.seed(seed)
        np.random.seed(seed)  # noqa: NPY002
        torch.manual_seed(seed)

        fold = seed_offset % cfg.cv.folds
        dm = _data_module_cls(cfg)(
            cfg.dataset.name,
            root=cfg.dataset.root,
            fold=fold,
            to_undirected=not cfg.dataset.directed,
        )
        dm.setup()

        primary_key = f"val_{dm.info.metric}"
        monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else primary_key
        mode = "min" if cfg.optim.stop_strategy == "loss" else "max"
        module = select_module_cls(cfg)(cfg, dm.info)

        # Evaluate best-val-epoch weights, not last-epoch weights: with
        # patience 200 the final model is far past its validation peak.
        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                filename="best",
            )
            callbacks: list = [
                EarlyStopping(
                    monitor=monitor,
                    patience=cfg.optim.early_stopping,
                    mode=mode,
                ),
                ckpt_cb,
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
                logger=False,
                log_every_n_steps=1,
            )
            start = start_resource_tracking(optuna_cfg.cuda)
            trainer.fit(module, dm)
            perf = resource_metrics(start, optuna_cfg.cuda, trainer.current_epoch)

            val_result = trainer.validate(module, dm, ckpt_path="best", verbose=False)[
                0
            ]
            # Reference protocol: also record the TEST metrics at the
            # best-validation checkpoint for every trial (never used for
            # selection; the Optuna objective stays validation-based).
            test_result = trainer.test(module, dm, ckpt_path="best", verbose=False)[0]
        for key, value in {**val_result, **test_result, **perf}.items():
            seed_metrics[key].append(float(value))
        # .get + isinstance guard: a diverged fold may miss the primary key.
        val_metric = val_result.get(primary_key)
        val_txt = f"{val_metric:.4f}" if isinstance(val_metric, int | float) else "-"
        _console.print(
            f"[dim]trial {trial.number} fold"
            f" {seed_offset + 1}/{optuna_cfg.n_seeds_per_trial} seed {seed}:"
            f" {primary_key}={val_txt}"
            f" | {perf['epochs_trained']:.0f} ep"
            f" | {_fmt_duration(perf['fit_time_s'])}[/dim]"
        )

    # Aggregate all metrics across seeds and store as trial attributes for W&B;
    # std is always reported (0.0 for a single seed).
    aggregated: dict[str, float] = {}
    for key, values in seed_metrics.items():
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = (
            float(np.std(values)) if optuna_cfg.n_seeds_per_trial > 1 else 0.0
        )
        trial.set_user_attr(f"mean_{key}", aggregated[f"mean_{key}"])
        trial.set_user_attr(f"std_{key}", aggregated[f"std_{key}"])

    # Log aggregate metrics only; identity lives in run config. wandb.log writes
    # the summary, which pruned/failed trials never reach (-> grouped as null).
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(aggregated | {"n_seeds": optuna_cfg.n_seeds_per_trial})
    except ImportError:
        pass

    primary_values = seed_metrics.get(primary_key, [0.0])
    mean = float(np.mean(primary_values))
    std = float(np.std(primary_values)) if optuna_cfg.n_seeds_per_trial > 1 else 0.0
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

    # The metric name lives in the registry; do not load the dataset for it.
    from exp.registries.datasets import dataset_registry

    metric = dataset_registry.get(base_cfg.dataset.name).metric  # pragma: no cover

    kwargs: dict = {  # pragma: no cover
        "metric_name": f"val_{metric}",
        # One wandb run per Optuna trial: every trial gets its own row in the
        # project table, grouped under the study name.
        "as_multirun": True,
        "wandb_kwargs": {
            "project": sweep_cfg.config.wandb_project,
            "entity": sweep_cfg.config.wandb_entity,
            "group": sweep_cfg.config.study_name,
            # Identity in the run config AT CREATION: W&B snapshots the group
            # key when the run is born, so a mid-run config.update lands too
            # late and the table buckets the run as null. transport_type is the
            # fixed variant; the per-trial stamp overrides it if variant is
            # swept.
            "config": {
                "model": sweep_cfg.model,
                "dataset": base_cfg.dataset.name,
                "transport_type": base_cfg.model.variant,
            },
        },
    }
    return [WeightsAndBiasesCallback(**kwargs)]  # pragma: no cover


# ---------------------------------------------------------------------------
# Core logic (testable without CLI)
# ---------------------------------------------------------------------------


def sweep(
    yaml_path: Path,
    preset: str | None = None,
) -> None:
    """Run a YAML-driven Optuna hyperparameter sweep."""
    _silence_training_noise()
    raw = yaml.safe_load(yaml_path.read_text())
    sweep_cfg = SweepConfig.model_validate(raw)
    base_cfg = preset_registry.get_or_default(preset)

    if sweep_cfg.dataset is not None:
        base_cfg = dataclasses.replace(
            base_cfg,
            dataset=DatasetConfig(
                name=sweep_cfg.dataset.name,
                root=sweep_cfg.dataset.root,
                directed=sweep_cfg.dataset.directed,
            ),
        )

    setup_torch(precision="high", seed=sweep_cfg.config.seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    # Seeded sampler: colleagues reproduce the same trial sequence.
    sampler = optuna.samplers.TPESampler(seed=sweep_cfg.config.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=sweep_cfg.config.study_name,
        storage=sweep_cfg.config.storage,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    _console.print(
        f"[bold]sweep[/bold] study={sweep_cfg.config.study_name!r}"
        f" model={sweep_cfg.model} variant={base_cfg.model.variant}"
        f" dataset={base_cfg.dataset.name}"
        f" trials={sweep_cfg.config.n_trials}"
        f" seeds/trial={sweep_cfg.config.n_seeds_per_trial}"
        f" storage={sweep_cfg.config.storage or 'in-memory'}"
    )

    # Shipped sweeps are named after their variant, which only the preset
    # actually sets; a wrong --preset silently trains (and logs to W&B) a
    # different transport_type than the study name claims.
    stem = yaml_path.stem
    if (
        stem in _KNOWN_VARIANTS
        and "variant" not in sweep_cfg.search_space
        and stem != base_cfg.model.variant
    ):
        _console.print(
            f"[bold red]WARNING[/bold red] {yaml_path.name} looks like a"
            f" {stem!r} sweep, but preset {preset!r} sets"
            f" variant={base_cfg.model.variant!r}: training and the W&B"
            f" transport_type will use {base_cfg.model.variant!r}."
            f" Pass the preset documented in the YAML header if unintended."
        )

    wandb_callbacks = (
        _make_wandb_callbacks(base_cfg, sweep_cfg)
        if sweep_cfg.config.wandb_project
        else []
    )

    def objective(trial: optuna.Trial) -> float:
        return _run_trial(trial, sweep_cfg, base_cfg)

    if wandb_callbacks:
        # track_in_wandb makes each trial's own run active inside the
        # objective, so metrics logged there land on that trial's row.
        objective = wandb_callbacks[0].track_in_wandb()(objective)

    # Diverged trials (non-finite maps -> LinAlgError/RuntimeError) are marked
    # FAILED instead of killing the whole sweep.
    # MaxTrialsCallback caps the study's total trial count across resumes and
    # workers; a bare n_trials would add n_trials more on every relaunch.
    budget_states = (
        (TrialState.COMPLETE,)
        if sweep_cfg.config.budget_states == "complete"
        else (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    )
    study.optimize(
        objective,
        callbacks=[
            *wandb_callbacks,
            _SweepProgress(sweep_cfg.config.n_trials),
            MaxTrialsCallback(sweep_cfg.config.n_trials, states=budget_states),
        ],
        catch=(RuntimeError,),
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

    _console.print("\n[bold]Final test evaluation — best config, 10 folds[/bold]")
    test_results = _run_final_test(sweep_cfg, base_cfg, best.params, n_seeds=10)
    _console.print("  [bold]test results:[/bold]")
    for key in sorted(k for k in test_results if k.startswith("mean_")):
        metric = key[len("mean_") :]
        std_val = test_results.get(f"std_{metric}", 0.0)
        _console.print(f"    {metric}: {test_results[key]:.4f} ± {std_val:.4f}")

    if sweep_cfg.config.wandb_project:
        _log_final_test_to_wandb(
            sweep_cfg,
            best.params,
            test_results,
            n_seeds=10,
            dataset=base_cfg.dataset.name,
            transport_type=str(best.params.get("variant", base_cfg.model.variant)),
        )


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


def _run_final_test(
    sweep_cfg: SweepConfig,
    base_cfg: Config,
    best_params: dict,
    n_seeds: int = 10,
) -> dict[str, float]:
    """Retrain best config from scratch over n_seeds and evaluate on test split."""
    cfg = _build_cfg(base_cfg, sweep_cfg.model, best_params)
    optuna_cfg = sweep_cfg.config
    seed_metrics: dict[str, list[float]] = defaultdict(list)

    for seed_offset in range(n_seeds):
        seed = optuna_cfg.seed + seed_offset
        random.seed(seed)
        np.random.seed(seed)  # noqa: NPY002
        torch.manual_seed(seed)

        fold = seed_offset % cfg.cv.folds
        dm = _data_module_cls(cfg)(
            cfg.dataset.name,
            root=cfg.dataset.root,
            fold=fold,
            to_undirected=not cfg.dataset.directed,
        )
        dm.setup()

        monitor = (
            "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{dm.info.metric}"
        )
        mode = "min" if cfg.optim.stop_strategy == "loss" else "max"
        module = select_module_cls(cfg)(cfg, dm.info)

        # Test the best-val-epoch checkpoint, mirroring the reference protocol
        # (test accuracy is read at the epoch with the best validation score).
        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                filename="best",
            )
            trainer = Trainer(
                max_epochs=cfg.optim.epochs,
                callbacks=[
                    EarlyStopping(
                        monitor=monitor,
                        patience=cfg.optim.early_stopping,
                        mode=mode,
                    ),
                    ckpt_cb,
                ],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=[optuna_cfg.cuda] if torch.cuda.is_available() else "auto",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                log_every_n_steps=1,
            )
            start = start_resource_tracking(optuna_cfg.cuda)
            trainer.fit(module, dm)
            perf = resource_metrics(start, optuna_cfg.cuda, trainer.current_epoch)
            test_result = trainer.test(module, dm, ckpt_path="best", verbose=False)[0]
        for key, value in {**test_result, **perf}.items():
            seed_metrics[key].append(float(value))

    aggregated: dict[str, float] = {}
    for key, values in seed_metrics.items():
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


def _log_final_test_to_wandb(
    sweep_cfg: SweepConfig,
    best_params: dict,
    results: dict[str, float],
    n_seeds: int,
    dataset: str,
    transport_type: str,
) -> None:
    try:
        import wandb
    except ImportError:
        _console.print(
            "[yellow]wandb not installed; skipping W&B logging for final test.[/yellow]"
        )
        return

    run = wandb.init(
        project=sweep_cfg.config.wandb_project,
        entity=sweep_cfg.config.wandb_entity,
        name=f"{sweep_cfg.config.study_name}_final_test",
        config={
            **best_params,
            "n_seeds": n_seeds,
            "dataset": dataset,
            "model": sweep_cfg.model,
            "transport_type": transport_type,
        },
        job_type="final_test",
    )
    wandb.log(
        results
        | {
            "dataset": dataset,
            "model": sweep_cfg.model,
            "transport_type": transport_type,
        }
    )
    run.finish()  # type: ignore[union-attr]


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


if __name__ == "__main__":  # pragma: no cover
    tyro.cli(main)
