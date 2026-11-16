"""Validation-selected HSNN studies with per-trial and final test reporting."""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from rich.console import Console

from exp.config import Config, ModelConfig, ModelType
from exp.datasets.hypergraph import artifact_path, ensure_hypergraph, is_hypergraph
from exp.dhsnn_run import run as run_directed
from exp.hsnn_run import load_provenance, run, source_hashes
from exp.registries.presets import preset_registry
from exp.sweeps.models import SweepConfig

_console = Console()
_FIELDS = {
    "variant",
    "stalk_dim",
    "hidden_dim",
    "num_layers",
    "rank",
    "orth_strategy",
    "hsnn_residual",
    "hsnn_degree_regularization",
    "input_dropout",
    "dropout",
    "lr",
    "weight_decay",
    "epochs",
    "early_stopping",
    "dhsnn_q",
    "dhsnn_activation",
}
_FIELDS.update(
    f.name for f in dataclasses.fields(ModelConfig) if f.name.startswith("dhsnn_")
)


def _final_test_complete(folder: Path) -> bool:
    """Accept only a readable, finite test result for all ten distinct splits."""
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        metric = summary["metric"]
        rows = summary["runs"]
        if (
            summary.get("test_evaluated") is not True
            or metric not in ("accuracy", "f1")
            or len(rows) != 10
            or {row["split_row"] for row in rows} != set(range(10))
        ):
            return False
        return all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            for row in rows
            for value in (row[f"best_val_{metric}"], row["test"][metric])
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def export_best(study: optuna.Study, base: Config, folder: Path) -> None:
    """Export validation ranking and an explicit, unexecuted final-test command."""
    complete = study.get_trials(states=(TrialState.COMPLETE,))
    with (folder / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "trial",
                "state",
                "objective",
                "params",
                "artifacts",
                "validation_mean",
                "validation_std",
                "test_mean",
                "test_std",
            ]
        )
        for trial in study.trials:
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    trial.value,
                    json.dumps(trial.params),
                    trial.user_attrs.get("output"),
                    *(
                        trial.user_attrs.get(name)
                        for name in (
                            "validation_mean",
                            "validation_std",
                            "test_mean",
                            "test_std",
                        )
                    ),
                ]
            )
    if not complete:
        return
    from exp.sweeps.sweep import _build_cfg

    best = study.best_trial
    cfg = copy.deepcopy(_build_cfg(base, base.model.type.value, best.params))
    cfg.hypergraph.output = str(folder / f"final-trial-{best.number:05d}")
    cfg.hypergraph.evaluate_test = True
    cfg.cv.folds = 10
    (folder / "best.json").write_text(
        json.dumps(
            {
                "trial": best.number,
                "validation_objective": best.value,
                "validation_mean": best.user_attrs["validation_mean"],
                "validation_std": best.user_attrs["validation_std"],
                "config": dataclasses.asdict(cfg),
                "test_evaluated_during_search": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "exp.cli",
        "run",
        "--preset",
        ("dhsnn_cora" if base.model.type == ModelType.DHSNN else "hsnn_cora")
        if base.dataset.name == "cora_hypergraph"
        else base.dataset.name,
        "--model.type",
        base.model.type.value,
        "--dataset.root",
        cfg.dataset.root,
        "--hardware.cuda",
        str(cfg.hardware.cuda),
        "--cv.seed",
        str(cfg.cv.seed),
        "--cv.folds",
        "10",
        "--optim.epochs",
        str(cfg.optim.epochs),
        "--optim.early-stopping",
        str(cfg.optim.early_stopping),
        "--hypergraph.output",
        cfg.hypergraph.output,
        "--hypergraph.evaluate-test",
    ]
    for group, names in (
        (
            "model",
            (
                "variant",
                "stalk_dim",
                "hidden_dim",
                "num_layers",
                "rank",
                "orth_strategy",
                "hsnn_residual",
                "hsnn_degree_regularization",
                "dhsnn_q",
                "dhsnn_activation",
            ),
        ),
        ("reg", ("input_dropout", "dropout")),
        ("optim", ("lr", "weight_decay")),
    ):
        if group == "model" and base.model.type == ModelType.DHSNN:
            names = tuple(
                dict.fromkeys(
                    (
                        *names,
                        *(
                            f.name
                            for f in dataclasses.fields(ModelConfig)
                            if f.name.startswith("dhsnn_")
                        ),
                    )
                )
            )
        for name in names:
            value = getattr(getattr(cfg, group), name)
            flag = f"--{group}.{name.replace('_', '-')}"
            if isinstance(value, bool):
                command.append(
                    flag if value else f"--{group}.no-{name.replace('_', '-')}"
                )
            else:
                command.extend((flag, str(value)))
    (folder / "final-test.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "# Run from the project root after freezing the study.\n"
        + shlex.join(command)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (folder / "final-test-command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )


def sweep(spec: SweepConfig, preset: str | None = None) -> None:
    """Select by validation; record test scores without using them for selection."""
    from exp.sweeps.sweep import _build_cfg, _suggest, _SweepProgress

    # Both registry entries and the common builder share nested run controls.
    base = copy.deepcopy(preset_registry.get_or_default(preset))
    expected = ModelType(spec.model)
    if expected not in (ModelType.HSNN, ModelType.DHSNN):
        raise ValueError("Hypergraph sweeps require hsnn or dhsnn")
    if spec.dataset is not None:
        base.dataset = dataclasses.replace(base.dataset, **spec.dataset.model_dump())
    if not is_hypergraph(base.dataset.name):
        raise ValueError(
            "HSNN/DHSNN sweeps require the corresponding hypergraph preset"
        )
    base.model.type = expected
    base.dataset.directed = expected == ModelType.DHSNN
    if expected == ModelType.HSNN and base.dataset.directed:
        raise ValueError("HSNN sweeps require undirected hypergraph data")
    unknown = set(spec.search_space) - _FIELDS
    if unknown:
        raise ValueError(f"Unsupported HSNN search fields: {sorted(unknown)}")
    options = spec.config
    if options.n_trials < 1 or not 1 <= options.n_seeds_per_trial <= 10:
        raise ValueError("Positive trial budget and 1..10 frozen splits are required")
    if not math.isfinite(options.std_weight) or options.std_weight < 0:
        raise ValueError("std_weight must be finite and nonnegative")
    if options.wandb_project or options.wandb_entity:
        raise ValueError("HSNN studies save local artifacts; W&B is unsupported")
    if not options.storage or not options.storage.startswith("sqlite:///"):
        raise ValueError(
            "HSNN requires a local sqlite:/// storage for resumable trials"
        )
    db = Path(options.storage.removeprefix("sqlite:///"))
    db.parent.mkdir(parents=True, exist_ok=True)
    folder = db.parent / f"{db.stem}-artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    base.hardware.cuda = options.cuda
    base.cv.seed = options.seed
    base.cv.folds = options.n_seeds_per_trial
    base.hypergraph.evaluate_test = True
    base.hypergraph.output = None
    ensure_hypergraph(base.dataset.name, base.dataset.root)
    provenance = load_provenance(artifact_path(base.dataset.name, base.dataset.root))
    protocol = {
        "base_config": dataclasses.asdict(base),
        "search_space": {k: v.model_dump() for k, v in spec.search_space.items()},
        "std_weight": options.std_weight,
        "budget_states": options.budget_states,
        "data": provenance,
        "source_sha256": source_hashes(),
        "sweep_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    signature = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    study = optuna.create_study(
        study_name=options.study_name,
        storage=options.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=options.seed),
        pruner=optuna.pruners.NopPruner(),
    )
    previous = study.user_attrs.get("protocol_signature")
    if previous != signature and (previous is not None or study.trials):
        raise ValueError(
            "Study protocol/code/data changed; use a new storage/study name"
        )
    study.set_user_attr("protocol_signature", signature)
    (folder / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: _suggest(trial, name, value)
            for name, value in spec.search_space.items()
        }
        cfg = copy.deepcopy(_build_cfg(base, spec.model, params))
        cfg.hypergraph.evaluate_test = True
        cfg.hypergraph.output = str(folder / f"trial-{trial.number:05d}")
        trial.set_user_attr("output", cfg.hypergraph.output)
        try:
            runner = run_directed if expected == ModelType.DHSNN else run
            test_values = np.asarray(runner(cfg))
            # Runners return test scores when evaluation is enabled. Read the
            # validation scores at the selected checkpoints separately so test
            # reporting cannot change Optuna's selection objective.
            summary = json.loads(
                (Path(cfg.hypergraph.output) / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            metric = summary["metric"]
            values = np.asarray([row[f"best_val_{metric}"] for row in summary["runs"]])
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        if (
            values.shape != (options.n_seeds_per_trial,)
            or not np.isfinite(values).all()
            or test_values.shape != values.shape
            or not np.isfinite(test_values).all()
        ):
            raise RuntimeError("Invalid validation scores")
        mean, std = float(values.mean()), float(values.std())
        trial.set_user_attr("validation_mean", mean)
        trial.set_user_attr("validation_std", std)
        trial.set_user_attr("test_mean", float(test_values.mean()))
        trial.set_user_attr("test_std", float(test_values.std()))
        return mean - options.std_weight * std

    # A restarted process gets a seeded sampler, not the previous sampler's
    # exact RNG state; trial records and protocol remain persistent.
    budget_states = (
        (TrialState.COMPLETE,)
        if options.budget_states == "complete"
        else (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    )
    finished = len(study.get_trials(states=budget_states))
    remaining = max(0, options.n_trials - finished)
    _console.print(
        f"HSNN sweep: {options.study_name}; {finished}/{options.n_trials} finished; "
        f"{options.n_seeds_per_trial} splits/trial; test recorded, validation selection"
    )
    try:
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining if options.budget_states == "finished" else None,
                n_jobs=1,
                gc_after_trial=True,
                catch=(RuntimeError,),
                callbacks=[
                    _SweepProgress(options.n_trials),
                    optuna.study.MaxTrialsCallback(
                        options.n_trials, states=budget_states
                    ),
                    lambda study, trial: export_best(study, base, folder),
                ],
            )
    finally:
        export_best(study, base, folder)
    if not study.get_trials(states=(TrialState.COMPLETE,)):
        raise RuntimeError("No successful HSNN trial; inspect trial logs")
    _console.print(
        f"Best validation objective: {study.best_value:.4%}; "
        f"trial {study.best_trial.number}"
    )
    _console.print(f"Results: {folder}; final test command: {folder / 'final-test.sh'}")
    # Match graph sweeps: retrain the validation-selected configuration on ten
    # splits and evaluate test metrics automatically after search completes.
    final_cfg = copy.deepcopy(_build_cfg(base, spec.model, study.best_trial.params))
    final_cfg.cv.folds = 10
    final_cfg.hypergraph.evaluate_test = True
    final_output = folder / f"final-trial-{study.best_trial.number:05d}"
    final_cfg.hypergraph.output = str(final_output)
    completed = [final_output, *sorted(folder.glob(f"{final_output.name}-retry-*"))]
    if not any(_final_test_complete(path) for path in completed):
        attempt = 1
        while Path(final_cfg.hypergraph.output).exists():
            final_cfg.hypergraph.output = str(final_output) + f"-retry-{attempt}"
            attempt += 1
        runner = run_directed if expected == ModelType.DHSNN else run
        _console.print("Final test evaluation — best config, 10 folds")
        runner(final_cfg)
        if not _final_test_complete(Path(final_cfg.hypergraph.output)):
            raise RuntimeError("Final test did not produce ten valid split results")
