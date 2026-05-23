# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Generate or download pre-computed 60/20/20 train/val/test splits.

Two sources are supported:

canonical (default)
    Downloads the official Geom-GCN (Pei et al. 2020) splits from GitHub.
    These are the same splits used in the original NSD paper and most
    heterophily benchmarks. 10 folds per dataset.

generate
    Creates 10 stratified splits locally using StratifiedShuffleSplit.
    Useful as a fallback when offline or for datasets not covered by
    the canonical repository.

Files are saved to exp/splits/ as:
    exp/splits/{name}_split_0.6_0.2_{fold}.npz

Each .npz contains three 1-D boolean arrays of length N:
    train_mask, val_mask, test_mask

Usage
-----
    # Via the unified CLI (recommended):
    sheaf splits
    sheaf splits --datasets cora citeseer texas
    sheaf splits --source generate
    sheaf splits --root /data/pyg --splits-dir /data/splits
    sheaf splits --folds 5
    sheaf splits --overwrite

    # Direct module invocation:
    python -m exp.gen_splits
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import tyro
from rich.console import Console
from sklearn.model_selection import StratifiedShuffleSplit

from exp.data import load_dataset
from exp.registries import dataset_registry

_console = Console()

_DEFAULT_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
_N_FOLDS = 10
_TRAIN_RATIO = 0.6
_VAL_RATIO = 0.2

_GEOM_GCN_BASE = "https://github.com/graphdml-uiuc-jlu/geom-gcn/raw/master/splits"


def _npz_split_datasets() -> frozenset[str]:
    return frozenset(
        name
        for name in dataset_registry.list_keys()
        if dataset_registry.get(name).split_type == "npz_file"
    )


def _canonical_url(name: str, fold: int) -> str:
    return f"{_GEOM_GCN_BASE}/{name}_split_0.6_0.2_{fold}.npz"


@dataclass
class SplitsConfig:
    """Configuration for downloading or generating dataset splits."""

    datasets: list[str] = field(default_factory=lambda: sorted(_npz_split_datasets()))
    source: Literal["canonical", "generate"] = "canonical"
    root: str = "exp/data"
    splits_dir: str = _DEFAULT_SPLITS_DIR
    folds: int = _N_FOLDS
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Core logic (testable without CLI)
# ---------------------------------------------------------------------------


def download_canonical_splits(
    name: str,
    splits_dir: str = _DEFAULT_SPLITS_DIR,
    n_folds: int = _N_FOLDS,
    overwrite: bool = False,
) -> None:
    """Download the official Geom-GCN splits for *name* from GitHub.

    Skips any fold whose file already exists unless *overwrite* is True.
    """
    _console.print(f"[bold][{name}][/bold] Downloading canonical Geom-GCN splits...")
    os.makedirs(splits_dir, exist_ok=True)

    for fold in range(n_folds):
        out_path = os.path.join(splits_dir, f"{name}_split_0.6_0.2_{fold}.npz")

        if os.path.exists(out_path) and not overwrite:
            _console.print(f"  fold {fold:2d}: already exists, skipping.")
            continue

        url = _canonical_url(name, fold)
        try:
            urllib.request.urlretrieve(url, out_path)

            arr = np.load(out_path)
            train_n = int(arr["train_mask"].sum())
            val_n = int(arr["val_mask"].sum())
            test_n = int(arr["test_mask"].sum())
            _console.print(
                f"  fold {fold:2d}: "
                f"train={train_n:4d}  val={val_n:4d}  test={test_n:4d}  "
                f"[dim]<- {url}[/dim]"
            )
        except Exception as exc:
            if os.path.exists(out_path):
                os.remove(out_path)
            raise RuntimeError(
                f"Failed to download {url}: {exc}\n"
                "Tip: run with --source generate to create splits locally."
            ) from exc

    _console.print(f"[bold][{name}][/bold] Done.\n")


def generate_splits(
    name: str,
    root: str = "exp/data",
    splits_dir: str = _DEFAULT_SPLITS_DIR,
    n_folds: int = _N_FOLDS,
    overwrite: bool = False,
) -> None:
    """Generate and save n_folds stratified 60/20/20 splits for *name*.

    Skips any fold whose file already exists unless *overwrite* is True.
    """
    _console.print(f"[bold][{name}][/bold] Loading dataset...")
    data, info = load_dataset(name, root=root)
    assert isinstance(data.y, torch.Tensor)
    labels = data.y.numpy()
    n = int(labels.shape[0])

    os.makedirs(splits_dir, exist_ok=True)

    for fold in range(n_folds):
        out_path = os.path.join(splits_dir, f"{name}_split_0.6_0.2_{fold}.npz")

        if os.path.exists(out_path) and not overwrite:
            _console.print(f"  fold {fold:2d}: already exists, skipping.")
            continue

        sss1 = StratifiedShuffleSplit(
            n_splits=1, train_size=_TRAIN_RATIO, random_state=fold
        )
        train_idx, rest_idx = next(sss1.split(np.zeros(n), labels))

        val_of_rest = _VAL_RATIO / (1.0 - _TRAIN_RATIO)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, train_size=val_of_rest, random_state=fold
        )
        val_local, test_local = next(
            sss2.split(np.zeros(len(rest_idx)), labels[rest_idx])
        )
        val_idx = rest_idx[val_local]
        test_idx = rest_idx[test_local]

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        np.savez(
            out_path, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        _console.print(
            f"  fold {fold:2d}: "
            f"train={train_mask.sum():4d}  val={val_mask.sum():4d}  "
            f"test={test_mask.sum():4d}  [dim]-> {out_path}[/dim]"
        )

    _console.print(f"[bold][{name}][/bold] Done.\n")


def splits(cfg: SplitsConfig) -> None:
    """Process splits for all requested datasets according to *cfg*."""
    os.makedirs(cfg.splits_dir, exist_ok=True)
    valid = _npz_split_datasets()

    for name in cfg.datasets:
        if name not in valid:
            _console.print(
                f"[yellow]Warning:[/yellow] '{name}' does not use NPZ splits"
                " — skipping."
            )
            continue

        if cfg.source == "canonical":
            download_canonical_splits(
                name,
                splits_dir=cfg.splits_dir,
                n_folds=cfg.folds,
                overwrite=cfg.overwrite,
            )
        else:
            generate_splits(
                name,
                root=cfg.root,
                splits_dir=cfg.splits_dir,
                n_folds=cfg.folds,
                overwrite=cfg.overwrite,
            )

    _console.print("[green]All requested splits processed.[/green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python -m exp.gen_splits``."""
    cfg = tyro.cli(SplitsConfig)
    splits(cfg)


if __name__ == "__main__":
    main()
