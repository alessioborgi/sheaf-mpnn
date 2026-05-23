# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Train / val / test split management for 10-fold cross-validation.

Two split strategies are used depending on the dataset:

``npz_file``
    Pre-computed 60 / 20 / 20 splits stored as
    ``exp/splits/<dataset>_split_0.6_0.2_<fold>.npz``.
    Each file has keys ``train_mask``, ``val_mask``, ``test_mask`` - 1-D boolean
    arrays of length N.  Used for: cora, citeseer, chameleon, squirrel,
    cornell, texas, film.

``pyg_mask``
    Fold column extracted from the multi-column masks that PyG attaches to
    HeterophilousGraphDataset, FilteredWikipediaDataset, etc.  The masks are
    ``(N, num_splits)`` boolean tensors; ``apply_split`` selects column *fold*.
    Used for: amazon_ratings, minesweeper, questions, roman_empire, tolokers,
    chameleon_filtered, squirrel_filtered.

The public function :func:`apply_split` dispatches automatically based on
:attr:`DatasetInfo.split_type`.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
from torch_geometric.data import Data

from exp.data import DatasetInfo

# Absolute path to the splits directory that lives next to this file.
_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")

# Canonical Geom-GCN splits repository (Pei et al. 2020, ICLR).
_GEOM_GCN_BASE = "https://github.com/graphdml-uiuc-jlu/geom-gcn/raw/master/splits"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_split(name: str, fold: int, path: str) -> None:
    """Download the canonical Geom-GCN split for *name* / *fold* to *path*."""
    url = f"{_GEOM_GCN_BASE}/{name}_split_0.6_0.2_{fold}.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[splits] Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Failed to download canonical split from {url}: {exc}\n"
            "Run `python -m exp.gen_splits --source generate` to build splits locally."
        ) from exc
    print(f"[splits] Saved -> {path}")


def _apply_npz_split(data: Data, name: str, fold: int) -> Data:
    """Load a pre-computed .npz split and stamp it onto *data*.

    If the file is absent it is downloaded automatically from the canonical
    Geom-GCN repository (Pei et al. 2020) before loading.
    """
    # Split files follow the fixed 60/20/20 naming scheme used by the benchmarks.
    path = os.path.join(_SPLITS_DIR, f"{name}_split_0.6_0.2_{fold}.npz")

    if not os.path.isfile(path):
        _download_split(name, fold, path)

    # Clone before attaching masks so callers keep the original dataset object intact.
    split = np.load(path)
    data = data.clone()
    data.train_mask = torch.from_numpy(split["train_mask"]).bool()
    data.val_mask = torch.from_numpy(split["val_mask"]).bool()
    data.test_mask = torch.from_numpy(split["test_mask"]).bool()
    return data


def _apply_pyg_mask_split(data: Data, fold: int) -> Data:
    """Select the *fold*-th column from multi-column PyG mask tensors."""
    data = data.clone()
    if data.train_mask.dim() == 1:
        # Dataset has a single split - return as-is (fold index is ignored)
        return data
    n_available = data.train_mask.size(1)
    col = fold % n_available  # wrap gracefully if fold >= n_available
    data.train_mask = data.train_mask[:, col].bool()
    data.val_mask = data.val_mask[:, col].bool()
    data.test_mask = data.test_mask[:, col].bool()
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_split(data: Data, info: DatasetInfo, fold: int) -> Data:
    """Return a *clone* of *data* with 1-D boolean masks for the given fold.

    Args:
        data: The full graph dataset (masks may be multi-column at this point).
        info: Metadata returned by :func:`~exp.data.load_dataset`.
        fold: Zero-based fold index in ``[0, info.num_splits)``.

    Returns:
        A cloned ``Data`` object whose ``train_mask``, ``val_mask``, and
        ``test_mask`` are 1-D boolean tensors of length N.
    """
    if info.split_type == "npz_file":
        return _apply_npz_split(data, info.name, fold)
    return _apply_pyg_mask_split(data, fold)
