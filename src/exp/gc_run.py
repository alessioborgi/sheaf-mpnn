# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Sheaf model training on TUDataset graph-classification benchmarks.

Datasets (pass via ``--dataset.name``):

- Heterogeneous: ``mutag``, ``proteins``, ``enzymes``, ``nci1``, ``nci109``,
  ``ptcmr``, ``dd``
- Homogeneous: ``collab``, ``imdb_b``, ``imdb_m``, ``reddit_b``, ``reddit_5k``

Usage::

    python -m exp.gc_run --dataset.name mutag --model.variant general
    python -m exp.gc_run --preset cora --dataset.name proteins  # borrow preset
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
from tqdm import tqdm

from exp.config import Config, ModelType
from exp.datasets.graph_classification import (
    GCHetSheafDataModule,
    GCSheafDataModule,
    gc_het_registry,
)
from exp.datasets.info import GCDatasetInfo
from exp.gc_module import GCLightningModule
from exp.registries.presets import preset_registry
from exp.run import finish_wandb_run
from sheaf_mpnn.utils import setup_torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataModule selector
# ---------------------------------------------------------------------------


def _make_dm(
    name: str,
    root: str,
    fold: int,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> GCSheafDataModule:
    """Return the right DataModule for the given dataset name."""
    if name.lower() in gc_het_registry:
        return GCHetSheafDataModule(
            name,
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    return GCSheafDataModule(
        name,
        root=root,
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


# ---------------------------------------------------------------------------
# Helpers (mirrors het_run.py / run.py)
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_config() -> Config:
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
                f"Unknown preset {preset_name!r}. Known: {', '.join(known)}"
            )
        default: Config | None = preset_registry.get(preset_name)
    else:
        default = None
    result = tyro.cli(Config, default=default)
    assert result is not None
    return result


def _make_logger(
    cfg: Config, info: GCDatasetInfo, fold: int, run_name: str
) -> _WandbLogger | bool:
    if not cfg.wandb.enabled:
        return False
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project=cfg.wandb.project or f"gc-nsd-{info.name}",
        entity=cfg.wandb.entity,
        name=f"{run_name}-fold{fold}",
        group=run_name,
        config=dataclasses.asdict(cfg),
    )


def _model_label(cfg: Config) -> str:
    return f"{cfg.model.type}-{cfg.model.variant}"


def _run_fold(
    cfg: Config,
    info: GCDatasetInfo,
    fold: int,
    monitor: str,
    ckpt_mode: str,
) -> float:
    run_name = (
        f"{_model_label(cfg)}-d{cfg.model.stalk_dim}"
        f"-h{cfg.model.hidden_dim}-L{cfg.model.num_layers}"
    )
    dm = _make_dm(
        cfg.dataset.name,
        cfg.dataset.root,
        fold,
        cfg.optim.batch_size,
        cfg.hardware.num_workers,
        cfg.hardware.pin_memory,
        cfg.hardware.persistent_workers,
    )
    dm.setup()
    module = GCLightningModule(cfg, info)
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
        try:
            trainer.fit(module, dm)
            [test_res] = trainer.test(module, dm, ckpt_path="best", verbose=False)
        finally:
            finish_wandb_run(logger)

    return float(test_res.get("test_acc", 0.0))


def _silence_third_party() -> None:
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.*")
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for TUDataset graph-classification experiments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _silence_third_party()
    cfg = _parse_config()
    setup_torch(precision="high", seed=cfg.cv.seed)

    # Load metadata from fold 0 to populate info once.
    dm_meta = _make_dm(
        cfg.dataset.name,
        cfg.dataset.root,
        fold=0,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        persistent_workers=cfg.hardware.persistent_workers,
    )
    dm_meta.setup()
    info = dm_meta.info
    assert isinstance(info, GCDatasetInfo)

    n_folds = min(cfg.cv.folds, info.num_splits)
    n_train, n_val, n_test = dm_meta.split_sizes
    total_graphs = n_train + n_val + n_test

    sep = "-" * 60
    log.info(sep)
    log.info(
        "Dataset  %s  graphs=%s  task=%s",
        f"{info.name:<16}",
        f"{total_graphs:,}",
        info.task,
    )
    log.info(
        "         F=%d  C=%d  node_types=%d  edge_types=%d  metric=%s",
        info.num_features,
        info.num_classes,
        info.num_node_types,
        info.num_edge_types,
        info.metric,
    )
    log.info(
        "Split    train=%s  val=%s  test=%s  (%d-fold CV)",
        f"{n_train:,}",
        f"{n_val:,}",
        f"{n_test:,}",
        n_folds,
    )
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
    log.info("%s", ModelSummary(GCLightningModule(cfg, info), max_depth=1))
    log.info(sep)

    monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else "val_acc"
    ckpt_mode = "min" if cfg.optim.stop_strategy == "loss" else "max"

    results: list[float] = []
    for fold in tqdm(range(n_folds), desc="CV folds", unit="fold"):
        _set_seed(cfg.cv.seed + fold)
        test_acc = _run_fold(cfg, info, fold, monitor, ckpt_mode)
        results.append(test_acc)
        log.info("  fold %2d | test acc: %.4f", fold, test_acc)

    arr = np.array(results)
    log.info("\n%s", sep)
    log.info("%s on %s [%d folds]", _model_label(cfg), info.name, len(results))
    log.info("Test  acc: %.2f +/- %.2f%%", arr.mean() * 100, arr.std() * 100)
    log.info(sep)


if __name__ == "__main__":  # pragma: no cover
    main()
