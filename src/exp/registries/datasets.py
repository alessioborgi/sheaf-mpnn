# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Dataset registry: metadata for all supported benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from exp.registries.base import Registry


@dataclass(frozen=True)
class DatasetEntry:
    """Static metadata for a benchmark dataset.

    Attributes:
        metric: Evaluation metric - ``"acc"`` (accuracy) or ``"roc_auc"``.
        split_type: How train/val/test splits are sourced.  ``"npz_file"``
            means pre-computed Geom-GCN splits live under ``exp/splits/``;
            ``"pyg_mask"`` means the PyG dataset ships its own masks.
        num_splits: Number of available train/val/test folds.
    """

    metric: Literal["acc", "roc_auc"]
    split_type: Literal["npz_file", "pyg_mask"]
    num_splits: int = 10


class DatasetRegistry(Registry[str, DatasetEntry]):
    """Registry of benchmark dataset metadata."""


# ---------------------------------------------------------------------------
# Registry instance  one entry per supported dataset
# ---------------------------------------------------------------------------

dataset_registry: DatasetRegistry = DatasetRegistry()

# Geom-GCN / WebKB / Actor datasets  use pre-computed NPZ splits.
dataset_registry.register("cora", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register("citeseer", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register("pubmed", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register(
    "chameleon", DatasetEntry(metric="acc", split_type="npz_file")
)
dataset_registry.register("squirrel", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register("cornell", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register("texas", DatasetEntry(metric="acc", split_type="npz_file"))
dataset_registry.register(
    "wisconsin", DatasetEntry(metric="acc", split_type="npz_file")
)
dataset_registry.register("film", DatasetEntry(metric="acc", split_type="npz_file"))

# Filtered Wikipedia variants  PyG ships masks, no NPZ splits.
dataset_registry.register(
    "chameleon_filtered", DatasetEntry(metric="acc", split_type="pyg_mask")
)
dataset_registry.register(
    "squirrel_filtered", DatasetEntry(metric="acc", split_type="pyg_mask")
)

# Platonov heterophilous datasets  PyG ships masks; binary ones use ROC-AUC.
dataset_registry.register(
    "amazon_ratings", DatasetEntry(metric="acc", split_type="pyg_mask")
)
dataset_registry.register(
    "roman_empire", DatasetEntry(metric="acc", split_type="pyg_mask")
)
dataset_registry.register(
    "minesweeper", DatasetEntry(metric="roc_auc", split_type="pyg_mask")
)
dataset_registry.register(
    "questions", DatasetEntry(metric="roc_auc", split_type="pyg_mask")
)
dataset_registry.register(
    "tolokers", DatasetEntry(metric="roc_auc", split_type="pyg_mask")
)

# OGB node-property datasets (Hu et al. 2020): one official split each.
# ogbn_proteins is multilabel (112 binary targets) scored by mean ROC-AUC.
dataset_registry.register(
    "ogbn_arxiv", DatasetEntry(metric="acc", split_type="pyg_mask", num_splits=1)
)
dataset_registry.register(
    "ogbn_products", DatasetEntry(metric="acc", split_type="pyg_mask", num_splits=1)
)
dataset_registry.register(
    "ogbn_proteins",
    DatasetEntry(metric="roc_auc", split_type="pyg_mask", num_splits=1),
)

# LINKX (Lim et al. 2021): five official fixed 50/25/25 splits.
dataset_registry.register(
    "pokec", DatasetEntry(metric="acc", split_type="pyg_mask", num_splits=5)
)

# Directed-only datasets (DSNN sources; see exp.datasets.directed / .synthetic).
dataset_registry.register("telegram", DatasetEntry(metric="acc", split_type="pyg_mask"))
for _alpha in ("0.05", "0.08", "0.1"):
    for _beta in ("0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"):
        dataset_registry.register(
            f"dataset_nodes500_alpha{_alpha}_beta{_beta}",
            DatasetEntry(metric="acc", split_type="pyg_mask"),
        )
