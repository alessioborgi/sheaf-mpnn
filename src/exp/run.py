# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Neural Sheaf Diffusion - 10-fold cross-validation experiment runner.

Usage
-----
    # Via the unified CLI (recommended):
    sheaf run --preset cora
    sheaf run --preset texas --model.num_layers 3

    # Direct module invocation:
    python -m exp.run --preset cora
"""

from __future__ import annotations

import dataclasses
import logging
import random
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.loggers import WandbLogger as _WandbLogger

import numpy as np
import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from exp.config import Config
from exp.data import DatasetInfo, SheafDataModule
from exp.module import SheafLightningModule
from exp.registries.presets import preset_registry
from sheaf_mpnn.utils import setup_torch

log = logging.getLogger(__name__)
_console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_config() -> Config:
    """Handle ``--preset <name>`` before handing the rest to tyro.

    The preset is stripped from ``sys.argv`` and used to set the ``default``
    argument of ``tyro.cli``, so every field can still be overridden.
    """
    argv = sys.argv[1:]
    preset_name: str | None = None
    clean: list[str] = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--preset":
            if i + 1 < len(argv):
                preset_name = argv[i + 1]
                i += 2
            else:
                i += 1
        elif a.startswith("--preset="):
            preset_name = a.split("=", 1)[1]
            i += 1
        else:
            clean.append(a)
            i += 1

    sys.argv = [sys.argv[0]] + clean
    if preset_name is not None:
        if preset_name not in preset_registry:
            known = sorted(preset_registry.list_keys())
            raise SystemExit(
                f"Unknown preset {preset_name!r}. "
                f"Run with --help to see available presets.\n"
                f"Known: {', '.join(known)}"
            )
        default: Config | None = preset_registry.get(preset_name)
    else:
        default = None
    result = tyro.cli(Config, default=default)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------


def _make_logger(
    cfg: Config, info: DatasetInfo, fold: int, run_name: str
) -> _WandbLogger | bool:
    if not cfg.wandb.enabled:
        return False
    from lightning.pytorch.loggers import WandbLogger  # lazy - wandb is optional

    return WandbLogger(
        project=cfg.wandb.project or f"nsd-{info.name}",
        entity=cfg.wandb.entity,
        name=f"{run_name}-fold{fold}",
        group=run_name,
        config=dataclasses.asdict(cfg),
    )


def _model_label(cfg: Config) -> str:
    """Short identifier for logging and W&B run names."""
    return f"{cfg.model.type}-{cfg.model.variant}"


def _run_fold(
    cfg: Config,
    info: DatasetInfo,
    fold: int,
    monitor: str,
    ckpt_mode: str,
) -> float:
    """Train and evaluate one cross-validation fold; return the test metric."""
    run_name = (
        f"{_model_label(cfg)}-d{cfg.model.stalk_dim}"
        f"-h{cfg.model.hidden_dim}-L{cfg.model.num_layers}"
    )
    dm = SheafDataModule(
        cfg.dataset.name,
        root=cfg.dataset.root,
        fold=fold,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        persistent_workers=cfg.hardware.persistent_workers,
    )
    module = SheafLightningModule(cfg, info)
    logger = _make_logger(cfg, info, fold, run_name)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor,
            mode=ckpt_mode,
            save_top_k=1,
            filename="best",
        )

        trainer = Trainer(
            max_epochs=cfg.optim.epochs,
            callbacks=[
                EarlyStopping(
                    monitor=monitor,
                    patience=cfg.optim.early_stopping,
                    mode=ckpt_mode,
                ),
                ckpt_cb,
            ],
            logger=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[cfg.hardware.cuda] if torch.cuda.is_available() else "auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        trainer.fit(module, dm)
        [test_res] = trainer.test(module, dm, ckpt_path="best", verbose=False)

    return float(test_res.get(f"test_{info.metric}", 0.0))


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def _display_startup(
    cfg: Config,
    info: DatasetInfo,
    n_train: int,
    n_val: int,
    n_test: int,
    avg_deg: float,
    n_folds: int,
) -> None:
    model_line = (
        f"{cfg.model.type.upper()}-{cfg.model.variant}  "
        f"d={cfg.model.stalk_dim}  hidden={cfg.model.hidden_dim}  "
        f"layers={cfg.model.num_layers}  alpha={cfg.model.alpha}  "
        f"dropout={cfg.reg.dropout}"
    )
    summary = str(ModelSummary(SheafLightningModule(cfg, info), max_depth=1))
    content = (
        f"[bold]Dataset[/bold]  {info.name}  "
        f"N={info.num_features:,}  E={info.num_features:,}  avg_deg={avg_deg:.1f}\n"
        f"         F={info.num_features}  C={info.num_classes}  metric={info.metric}\n"
        f"[bold]Split[/bold]    train={n_train:,}  val={n_val:,}  test={n_test:,}  "
        f"({n_folds}-fold CV)\n"
        f"[bold]Model[/bold]    {model_line}\n\n"
        f"{summary}"
    )
    _console.print(Panel(content, title="NSD Cross-Validation"))


def _display_results(cfg: Config, info: DatasetInfo, results: list[float]) -> None:
    scale = 100.0 if info.metric == "acc" else 1.0
    suffix = "%" if info.metric == "acc" else ""

    table = Table(title=f"{_model_label(cfg)} on {info.name} [{len(results)} folds]")
    table.add_column("Fold", justify="right", style="cyan")
    table.add_column(f"Test {info.metric}", justify="right")

    for fold, metric in enumerate(results):
        table.add_row(str(fold), f"{metric * scale:.4f}{suffix}")

    arr = np.array(results)
    table.add_section()
    table.add_row(
        "Mean ± Std",
        f"[bold]{arr.mean() * scale:.2f} ± {arr.std() * scale:.2f}{suffix}[/bold]",
    )
    _console.print(table)


# ---------------------------------------------------------------------------
# Main logic (testable without CLI)
# ---------------------------------------------------------------------------


def run(cfg: Config) -> list[float]:
    """Run 10-fold cross-validation; return per-fold test metrics."""
    _silence_third_party()
    setup_torch(precision="high", seed=cfg.cv.seed)

    dm_meta = SheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=0)
    dm_meta.setup()
    info = dm_meta.info
    n_folds = min(cfg.cv.folds, info.num_splits)

    n_train, n_val, n_test = dm_meta.split_sizes
    avg_deg = (dm_meta.num_edges * 2) / dm_meta.num_nodes

    _display_startup(cfg, info, n_train, n_val, n_test, avg_deg, n_folds)

    monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{info.metric}"
    ckpt_mode = "min" if cfg.optim.stop_strategy == "loss" else "max"

    results: list[float] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running folds...", total=n_folds)
        for fold in range(n_folds):
            _set_seed(cfg.cv.seed + fold)
            test_metric = _run_fold(cfg, info, fold, monitor, ckpt_mode)
            results.append(test_metric)
            progress.update(
                task,
                advance=1,
                description=f"fold {fold:2d}  {info.metric}={test_metric:.4f}",
            )

            if fold == 0 and info.metric == "acc" and test_metric < cfg.cv.min_acc:
                _console.print(
                    f"[yellow]fold 0 test {info.metric}={test_metric:.4f} "
                    f"< min_acc={cfg.cv.min_acc:.4f}; aborting run.[/yellow]"
                )
                break

    _display_results(cfg, info, results)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _silence_third_party() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")


def main() -> None:
    """Entry point for ``python -m exp.run``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _parse_config()
    run(cfg)


if __name__ == "__main__":
    main()
