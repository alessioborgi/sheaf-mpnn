# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Synthetic *directed* node-classification datasets (DSBM graphs).

These are the directed Stochastic Block Model graphs shipped with the DSNN
project as pickled ``torch_geometric.data.Data`` objects (one ``.pk`` file per
``(alpha, beta)`` configuration, e.g.
``dataset_nodes500_alpha0.05_beta0.05.pk``).

The ``.pk`` files are not part of the git tree; they are downloaded on first
use from this repository's ``dataset-assets-v1`` release (see
:mod:`exp.datasets.assets`) into ``<root>/synthetic`` and cached there.

The loading / processing mirrors DSNN (``exp/run.py``):

1. Read the pickled ``Data`` -- it carries only ``y`` and a *directed*
   ``edge_index`` (no node features).
2. Keep the directed ``edge_index`` aside; symmetrise a copy with
   :func:`~torch_geometric.utils.to_undirected` -- the symmetric graph is used
   for split generation and (by default) for the degree features.
3. Build per-class 60/20/20 train/val/test masks for ``num_splits`` folds,
   replicating ``torch_geometric_signed_directed.node_class_split``
   (``train_size_per_class=0.6, val_size_per_class=0.2``).
4. Set node features to the in-/out-degree pair via :func:`in_out_degree`.
5. Restore the *directed* ``edge_index`` so directed models see directionality
   through the message-passing structure.

The returned ``(Data, DatasetInfo)`` pair plugs straight into the same
``SheafDataModule`` / ``apply_split`` / ``SheafLightningModule`` pipeline used
by the undirected benchmarks: masks are stored as ``(N, num_splits)`` tensors
and ``split_type="pyg_mask"``, so ``apply_split`` just selects the fold column.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if TYPE_CHECKING:
    from exp.data import DatasetInfo

# DSBM configurations hosted on the dataset release: 500 nodes, three alpha
# levels, eight beta levels (``%g`` formatting, matching the DSNN filenames).
_KNOWN_SYNTHETIC: frozenset[str] = frozenset(
    f"dataset_nodes500_alpha{a:g}_beta{b:g}"
    for a in (0.05, 0.08, 0.1)
    for b in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
)

# Package-relative fallback location for the synthetic ``.pk`` files, so a
# symlink into a DSNN checkout keeps working regardless of the working
# directory.
_PACKAGE_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "data", "synthetic")
)

DegreeFeatureSource = Literal["undirected", "directed"]


def synthetic_dir(root: str = "exp/data") -> str:
    """Return the directory holding the synthetic ``.pk`` files.

    Prefers ``<root>/synthetic`` when it exists, otherwise falls back to the
    package-relative ``exp/data/synthetic`` directory.
    """
    candidate = os.path.join(root, "synthetic")
    if os.path.isdir(candidate):
        return candidate
    return _PACKAGE_DEFAULT_DIR


def _data_path(filename: str, root: str) -> str:
    """Path of a synthetic data file, downloading it on first use."""
    for d in (os.path.join(root, "synthetic"), _PACKAGE_DEFAULT_DIR):
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    from exp.datasets.assets import ensure_asset  # noqa: PLC0415

    return ensure_asset(filename, os.path.join(root, "synthetic", filename))


def is_synthetic(name: str, root: str = "exp/data") -> bool:
    """True if *name* is a synthetic ``.pk`` graph (hosted or on disk)."""
    return name in _KNOWN_SYNTHETIC or os.path.isfile(
        os.path.join(synthetic_dir(root), f"{name}.pk")
    )


def available_synthetic(root: str = "exp/data") -> list[str]:
    """List the synthetic dataset names (hosted plus any found under *root*)."""
    names = set(_KNOWN_SYNTHETIC)
    d = synthetic_dir(root)
    if os.path.isdir(d):
        names.update(f[:-3] for f in os.listdir(d) if f.endswith(".pk"))
    return sorted(names)


def in_out_degree(
    edge_index: torch.Tensor, num_nodes: int, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """In-/out-degree node features, following DSNN's ``in_out_degree``.

    Returns a ``(num_nodes, 2)`` float tensor whose columns are the in-degree
    and out-degree of each node under the (possibly directed) ``edge_index``.
    """
    ei = edge_index.cpu().numpy()
    if weight is None:
        vals = np.ones(ei.shape[1], dtype=np.float32)
    else:
        vals = weight.cpu().numpy()
    adj = coo_matrix(
        (vals, (ei[0], ei[1])), shape=(num_nodes, num_nodes), dtype=np.float32
    ).tocsr()
    out_degree = np.asarray(np.abs(adj).sum(axis=0)).reshape(-1)  # column sums
    in_degree = np.asarray(np.abs(adj).sum(axis=1)).reshape(-1)  # row sums
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def per_class_node_split(
    y: torch.Tensor,
    num_splits: int = 10,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class train/val/test masks, replicating ``node_class_split``.

    For every fold and every class, the class' nodes are shuffled and split into
    ``train_ratio`` / ``val_ratio`` / remainder (test).  Returns three boolean
    tensors of shape ``(num_nodes, num_splits)`` matching the ``pyg_mask``
    convention used elsewhere in this repo.
    """
    y_np = y.cpu().numpy()
    num_nodes = y_np.shape[0]
    classes = np.unique(y_np)

    train = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    val = torch.zeros(num_nodes, num_splits, dtype=torch.bool)
    test = torch.zeros(num_nodes, num_splits, dtype=torch.bool)

    for split in range(num_splits):
        rng = np.random.default_rng(seed + split)
        for c in classes:
            idx = np.where(y_np == c)[0]
            rng.shuffle(idx)
            n = idx.shape[0]
            n_train = round(train_ratio * n)
            n_val = round(val_ratio * n)
            train[idx[:n_train], split] = True
            val[idx[n_train : n_train + n_val], split] = True
            test[idx[n_train + n_val :], split] = True

    return train, val, test


def load_synthetic(
    name: str,
    root: str = "exp/data",
    num_splits: int = 10,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
    degree_features_from: DegreeFeatureSource = "undirected",
) -> tuple[Data, DatasetInfo]:
    """Load and process a synthetic directed graph DSNN-style.

    Args:
        name: Base name of the ``.pk`` file (without extension), e.g.
            ``"dataset_nodes500_alpha0.05_beta0.05"``.
        root: Base data directory; ``<root>/synthetic`` is searched first.
        num_splits: Number of train/val/test folds to generate.
        train_ratio: Per-class fraction assigned to training.
        val_ratio: Per-class fraction assigned to validation (rest is test).
        seed: Base RNG seed for split generation.
        degree_features_from: Graph used for the in-/out-degree features.
            ``"undirected"`` matches DSNN exactly (in == out degree);
            ``"directed"`` uses the original directed edges so the two feature
            columns differ.

    Returns:
        ``(data, info)`` ready for ``SheafDataModule`` / ``apply_split``.
    """
    # Local import keeps exp.data <-> exp.datasets.synthetic free of import cycles.
    from exp.data import DatasetInfo  # noqa: PLC0415

    if not is_synthetic(name, root):
        raise FileNotFoundError(
            f"Synthetic dataset {name!r} is not hosted and not on disk. "
            f"Available: {available_synthetic(root)}"
        )
    path = _data_path(f"{name}.pk", root)

    with open(path, "rb") as f:
        raw = pickle.load(f)

    assert isinstance(raw.y, torch.Tensor), "synthetic graph must carry labels"
    y = raw.y.long()
    num_nodes = int(y.numel())

    directed_edges = raw.edge_index.long()
    undirected_edges = to_undirected(directed_edges, num_nodes=num_nodes)

    train_mask, val_mask, test_mask = per_class_node_split(
        y,
        num_splits=num_splits,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    feat_edges = (
        undirected_edges if degree_features_from == "undirected" else directed_edges
    )
    x = in_out_degree(feat_edges, num_nodes)

    data = Data(
        x=x,
        edge_index=directed_edges,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.num_nodes = num_nodes

    info = DatasetInfo(
        name=name,
        num_features=int(x.size(1)),
        num_classes=int(y.max().item()) + 1,
        num_splits=num_splits,
        metric="acc",
        split_type="pyg_mask",
    )
    return data, info


__all__ = [
    "DegreeFeatureSource",
    "available_synthetic",
    "in_out_degree",
    "is_synthetic",
    "load_synthetic",
    "per_class_node_split",
    "synthetic_dir",
]
