# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Sheaf model training on HGB heterogeneous node-classification datasets.

This script provides the main entry point for running experiments on HGB datasets
(IMDB, DBLP, ACM, Freebase). It handles configuration parsing, data loading,
model initialization, and training using PyTorch Lightning.

Datasets (pass via ``--dataset.name``): ``imdb``, ``dblp``, ``acm``, ``freebase``

Usage
-----
    python -m exp.het_run --dataset.name dblp --model.variant general
    python -m exp.het_run --preset cora --dataset.name dblp  # borrow model preset
"""

from __future__ import annotations

import logging
import random
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.loggers import WandbLogger as _WandbLogger

import dataclasses

import numpy as np
import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from tqdm import tqdm

from exp.config import Config, ModelType
from exp.data import DatasetInfo
from exp.datasets.hgb import HetSheafDataModule
from exp.datasets.info import HetDatasetInfo
from exp.module import HetBayesSheafLightningModule, HetSheafLightningModule
from exp.registries.presets import preset_registry
from exp.run import finish_wandb_run
from sheaf_mpnn.utils import setup_torch

# Configure module-level logger
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with run.py)
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Ensure reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_config() -> Config:
    """Parse command-line arguments using tyro.

    Supports an optional '--preset' flag to load pre-defined model/optim configs
    from the project's preset registry.
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

    # tyro will parse the remaining arguments
    sys.argv = [sys.argv[0]] + clean
    if preset_name is not None:
        if preset_name not in preset_registry:
            known = sorted(preset_registry.list_keys())
            raise SystemExit(
                f"Unknown preset {preset_name!r}. Known: {', '.join(known)}"
            )
        default: Config | None = preset_registry.get(preset_name)
    else:
        default = None

    # Instantiate the structured Config class from CLI args
    result = tyro.cli(Config, default=default)
    assert result is not None
    return result


def _make_logger(
    cfg: Config, info: DatasetInfo, fold: int, run_name: str
) -> _WandbLogger | bool:
    """Initialize WandB logging if enabled in the config."""
    if not cfg.wandb.enabled:
        return False
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project=cfg.wandb.project or f"het-nsd-{info.name}",
        entity=cfg.wandb.entity,
        name=f"{run_name}-fold{fold}",
        group=run_name,
        config=dataclasses.asdict(cfg),
    )


def _module_cls(cfg: Config) -> type[HetSheafLightningModule]:
    """Lightning module class for the configured het model type.

    HET_BSNN needs the ELBO loss and ensembled evaluation of
    ``HetBayesSheafLightningModule``; every other model keeps the standard
    het module.
    """
    if cfg.model.type == ModelType.HET_BSNN:
        return HetBayesSheafLightningModule
    return HetSheafLightningModule


def _model_label(cfg: Config) -> str:
    """Generate a readable label for the model configuration."""
    return f"{cfg.model.type}-{cfg.model.variant}"


def _run_fold(
    cfg: Config,
    info: DatasetInfo,
    fold: int,
    monitor: str,
    ckpt_mode: str,
) -> float:
    """Execute a single training and evaluation run.

    HGB datasets typically have a single fixed split, so this helper is usually
    called only once.

    Args:
        cfg: Configuration object.
        info: Dataset metadata.
        fold: Fold index (ignored for HGB).
        monitor: Metric to monitor for early stopping/checkpointing.
        ckpt_mode: Whether to minimize or maximize the monitored metric.

    Returns:
        The score on the test set for this run.
    """
    run_name = (
        f"{_model_label(cfg)}-d{cfg.model.stalk_dim}"
        f"-h{cfg.model.hidden_dim}-L{cfg.model.num_layers}"
    )

    # Initialize data and Lightning module
    dm = HetSheafDataModule(
        cfg.dataset.name,
        root=cfg.dataset.root,
        fold=fold,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        persistent_workers=cfg.hardware.persistent_workers,
    )
    module = _module_cls(cfg)(cfg, info)
    logger = _make_logger(cfg, info, fold, run_name)

    # Use a temporary directory for checkpointing to keep the workspace clean
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

        # Train and then test using the best saved checkpoint
        try:
            trainer.fit(module, dm)
            [test_res] = trainer.test(module, dm, ckpt_path="best", verbose=False)
        finally:
            finish_wandb_run(logger)

    return float(test_res.get(f"test_{info.metric}", 0.0))


def _silence_third_party() -> None:
    """Mute noisy warnings from upstream libraries (PyG, Lightning)."""
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for HGB heterogeneous node-classification experiments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _silence_third_party()

    # 1. Parse configuration and setup global precision
    cfg = _parse_config()
    setup_torch(precision="high", seed=cfg.cv.seed)

    # 2. Inspect dataset to log metadata and verify dimensions
    dm_meta = HetSheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=0)
    dm_meta.setup()
    info = dm_meta.info
    assert isinstance(info, HetDatasetInfo)

    # HGB datasets provide standardized, official train/val/test splits to
    # enable fair comparison across models. Consequently, cross-validation
    # is not applicable, and the experiment is run on a single fixed split.
    n_folds = 1
    n_train, n_val, n_test = dm_meta.split_sizes

    # 3. Log experimental summary
    sep = "-" * 60
    log.info(sep)
    log.info(
        "Dataset  %s  N=%s  E=%s",
        f"{info.name:<16}",
        f"{dm_meta.num_nodes:,}",
        f"{dm_meta.num_edges:,}",
    )
    log.info(
        "         F=%d  C=%d  node_types=%d  edge_types=%d  task=%s  metric=%s",
        info.num_features,
        info.num_classes,
        info.num_node_types,
        info.num_edge_types,
        info.task,
        info.metric,
    )
    log.info(
        "Split    train=%s  val=%s  test=%s",
        f"{n_train:,}",
        f"{n_val:,}",
        f"{n_test:,}",
    )

    # Log model architecture details
    if cfg.model.type == ModelType.SHEAF_ATTN:
        log.info(
            "Model    SheafAttn-%s  d=%d  hidden=%d  layers=%d  heads=%d",
            cfg.model.variant,
            cfg.model.stalk_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            cfg.model.num_heads,
        )
    else:
        log.info(
            "Model    %s-%s  d=%d  hidden=%d  layers=%d  alpha=%s",
            cfg.model.type.upper(),
            cfg.model.variant,
            cfg.model.stalk_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            cfg.model.alpha,
        )
    log.info(sep)
    log.info("%s", ModelSummary(_module_cls(cfg)(cfg, info), max_depth=1))
    log.info(sep)

    # Determine monitoring strategy (minimizing loss vs maximizing accuracy)
    monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{info.metric}"
    ckpt_mode = "min" if cfg.optim.stop_strategy == "loss" else "max"

    # 4. Execute training runs
    results: list[float] = []
    for fold in tqdm(range(n_folds), desc="CV folds", unit="fold"):
        _set_seed(cfg.cv.seed + fold)
        test_metric = _run_fold(cfg, info, fold, monitor, ckpt_mode)
        results.append(test_metric)
        log.info("  fold %2d | test %s: %.4f", fold, info.metric, test_metric)

    # 5. Log final results
    arr = np.array(results)
    scale = 100.0 if info.metric == "acc" else 1.0
    log.info("\n%s", sep)
    log.info("%s on %s [%d fold]", _model_label(cfg), info.name, len(results))
    log.info(
        "Test  %s: %.2f%s",
        info.metric,
        arr.mean() * scale,
        "%" if info.metric == "acc" else "",
    )
    log.info(sep)


if __name__ == "__main__":  # pragma: no cover
    main()
