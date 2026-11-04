# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Real-world *directed* node-classification datasets from DSNN.

These are the directed Platonov-style heterophilous graphs shipped with the
DSNN project as ``.npz`` archives.  Each archive has
the keys ``node_features``, ``node_labels``, ``edges`` (a directed ``(E, 2)``
edge list), and ``train_masks`` / ``val_masks`` / ``test_masks`` of shape
``(num_splits, N)``.

The data files are not part of the git tree; they are downloaded on first use
from this repository's ``dataset-assets-v1`` release (see
:mod:`exp.datasets.assets`) into ``<root>/directed`` and cached there.

Processing mirrors DSNN's ``utils.dataset.Dataset`` (minus the DGL dependency):

* node features are used as-is (no SGC/identity/adjacency augmentation);
* self-loops are dropped from the edge list but the graph is **kept directed**
  -- it is *not* symmetrised with ``to_undirected`` -- so directed models see
  the original directionality;
* the ``(num_splits, N)`` masks are transposed to ``(N, num_splits)`` to match
  the ``pyg_mask`` convention used elsewhere in this repo;
* the evaluation metric follows DSNN: ROC-AUC for binary tasks, accuracy
  otherwise.

The returned ``(Data, DatasetInfo)`` plugs straight into the same
``SheafDataModule`` / ``apply_split`` / ``SheafLightningModule`` pipeline as the
undirected benchmarks.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops

if TYPE_CHECKING:
    from exp.data import DatasetInfo

# Directed citation graphs (Heterophily_and_oversmoothing pickles, kept directed
# by DSNN).  Maps the public dataset name to the on-disk pickle basename.
_CITATION_BASES: dict[str, str] = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
}

# Directed ``.npz`` archives hosted on the dataset release.
_NPZ_NAMES: frozenset[str] = frozenset(
    {
        "roman_empire",
        "amazon_ratings",
        "questions",
        "chameleon_filtered",
        "squirrel_filtered",
        "minesweeper",
        "tolokers",
    }
)

# Package-relative fallback location for the directed data files, so a symlink
# into a DSNN checkout keeps working regardless of the working directory.
_PACKAGE_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "data", "directed")
)


def directed_dir(root: str = "exp/data") -> str:
    """Return the directory holding the directed data files.

    Prefers ``<root>/directed`` when it exists, otherwise falls back to the
    package-relative ``exp/data/directed`` directory.
    """
    candidate = os.path.join(root, "directed")
    if os.path.isdir(candidate):
        return candidate
    return _PACKAGE_DEFAULT_DIR


def _data_path(filename: str, root: str) -> str:
    """Path of a directed data file, downloading it on first use."""
    for d in (os.path.join(root, "directed"), _PACKAGE_DEFAULT_DIR):
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    from exp.datasets.assets import ensure_asset  # noqa: PLC0415

    return ensure_asset(filename, os.path.join(root, "directed", filename))


def is_directed_npz(name: str, root: str = "exp/data") -> bool:
    """True if *name* is a directed ``.npz`` dataset (hosted or on disk)."""
    return name in _NPZ_NAMES or os.path.isfile(
        os.path.join(directed_dir(root), f"{name}.npz")
    )


def available_directed(root: str = "exp/data") -> list[str]:
    """List the directed dataset names (hosted plus any found under *root*)."""
    names = set(_NPZ_NAMES)
    d = directed_dir(root)
    if os.path.isdir(d):
        names.update(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))
    return sorted(names)


def load_directed_npz(name: str, root: str = "exp/data") -> tuple[Data, DatasetInfo]:
    """Load a directed ``.npz`` dataset DSNN-style.

    Args:
        name: Base name of the ``.npz`` file (without extension), e.g.
            ``"roman_empire_directed"``.
        root: Base data directory; ``<root>/directed`` is searched first.

    Returns:
        ``(data, info)`` ready for ``SheafDataModule`` / ``apply_split``.
    """
    # Local import keeps exp.data <-> exp.datasets.directed free of import cycles.
    from exp.data import DatasetInfo  # noqa: PLC0415

    if not is_directed_npz(name, root):
        raise FileNotFoundError(
            f"Directed dataset {name!r} is not hosted and not on disk. "
            f"Available: {available_directed(root)}"
        )
    path = _data_path(f"{name}.npz", root)

    npz = np.load(path)
    x = torch.from_numpy(npz["node_features"]).float()
    y = torch.from_numpy(npz["node_labels"]).long()
    num_nodes = int(x.size(0))

    # Directed edge list (E, 2) -> (2, E); drop self-loops, keep directionality.
    edge_index = torch.from_numpy(npz["edges"]).long().t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)

    # Raw masks are (num_splits, N); store as (N, num_splits) like the PyG sets.
    train_mask = torch.from_numpy(npz["train_masks"]).t().bool().contiguous()
    val_mask = torch.from_numpy(npz["val_masks"]).t().bool().contiguous()
    test_mask = torch.from_numpy(npz["test_masks"]).t().bool().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.num_nodes = num_nodes

    num_classes = int(y.max().item()) + 1
    # DSNN protocol: binary tasks are scored with ROC-AUC, the rest with accuracy.
    metric = "roc_auc" if num_classes == 2 else "acc"

    info = DatasetInfo(
        name=name,
        num_features=int(x.size(1)),
        num_classes=num_classes,
        num_splits=int(train_mask.size(1)),
        metric=metric,
        split_type="pyg_mask",
    )
    return data, info


# ---------------------------------------------------------------------------
# Directed citation graphs (cora / citeseer / pubmed)
# ---------------------------------------------------------------------------


def is_directed_citation(name: str, root: str = "exp/data") -> bool:
    """True if *name* is a directed citation graph (cora/citeseer/pubmed)."""
    return name in _CITATION_BASES


def _normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Row-normalise features (L1), matching PyG's ``T.NormalizeFeatures``."""
    row_sum = x.sum(dim=1, keepdim=True).clamp_min(1.0)
    return x / row_sum


def _citation_masks(
    base: str, num_nodes: int, y: torch.Tensor, num_splits: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (N, num_splits) masks for a directed citation graph.

    Uses the canonical Geom-GCN 48/32/20 splits when their node count matches
    (cora, pubmed).  Falls back to generated per-class random splits when it does
    not -- e.g. DSNN's directed citeseer has 3312 nodes vs the 3327-node Geom-GCN
    split, exactly the case that makes DSNN switch to a random split.
    """
    from exp.splits import _SPLITS_DIR, _download_split  # noqa: PLC0415

    train_cols, val_cols, test_cols = [], [], []
    fallback_reason: str | None = None
    for fold in range(num_splits):
        path = os.path.join(_SPLITS_DIR, f"{base}_split_0.6_0.2_{fold}.npz")
        if not os.path.isfile(path):
            try:
                _download_split(base, fold, path)
            except Exception as exc:
                fallback_reason = f"split download failed ({exc})"
                break
        split = np.load(path)
        if split["train_mask"].shape[0] != num_nodes:
            fallback_reason = (
                f"node-count mismatch ({split['train_mask'].shape[0]} in the "
                f"Geom-GCN split vs {num_nodes} in the directed graph)"
            )
            break
        train_cols.append(torch.from_numpy(split["train_mask"]).bool())
        val_cols.append(torch.from_numpy(split["val_mask"]).bool())
        test_cols.append(torch.from_numpy(split["test_mask"]).bool())

    if fallback_reason is None:
        return (
            torch.stack(train_cols, dim=1),
            torch.stack(val_cols, dim=1),
            torch.stack(test_cols, dim=1),
        )

    # Size mismatch / unavailable: generate per-class 60/20/20 splits instead.
    # Warn loudly -- generated splits follow a different evaluation protocol, so
    # results are not comparable with runs that used the canonical fixed splits.
    warnings.warn(
        f"{base}: canonical Geom-GCN splits unavailable -- {fallback_reason}; "
        "falling back to generated per-class 60/20/20 splits (results are not "
        "comparable with the fixed-split protocol).",
        stacklevel=3,
    )
    from exp.datasets.synthetic import per_class_node_split  # noqa: PLC0415

    return per_class_node_split(y, num_splits=num_splits, seed=0)


def load_directed_citation(
    name: str, root: str = "exp/data"
) -> tuple[Data, DatasetInfo]:
    """Load a directed citation graph (cora/citeseer/pubmed) DSNN-style.

    Reads the ``{base}.edges/.features/.labels.pickle`` files, keeps the edges
    directed (no ``to_undirected``), row-normalises features like DSNN's
    ``T.NormalizeFeatures``, and attaches Geom-GCN (or generated) 10-fold splits.
    """
    from exp.data import DatasetInfo  # noqa: PLC0415

    base = _CITATION_BASES[name]

    def _load_pickle(kind: str):
        # These legacy pickles embed scipy/numpy objects saved under an older
        # numpy; unpickling them emits a benign dtype deprecation warning.
        path = _data_path(f"{base}.{kind}.pickle", root)
        with open(path, "rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pickle.load(f, encoding="latin1")

    edges = _load_pickle("edges")
    features = _load_pickle("features")
    labels = _load_pickle("labels")

    dense = features.todense() if hasattr(features, "todense") else features
    x = _normalize_features(torch.as_tensor(np.asarray(dense), dtype=torch.float))
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long).view(-1)
    num_nodes = int(x.size(0))

    edge_arr = np.asarray(edges, dtype=np.int64)
    edge_index = torch.from_numpy(edge_arr.T).contiguous()
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)

    train_mask, val_mask, test_mask = _citation_masks(base, num_nodes, y)

    data = Data(
        x=x,
        edge_index=edge_index,
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
        num_splits=int(train_mask.size(1)),
        metric="acc",
        split_type="pyg_mask",
    )
    return data, info


# ---------------------------------------------------------------------------
# Telegram (torch_geometric_signed_directed)
# ---------------------------------------------------------------------------


def load_telegram(root: str = "exp/data") -> tuple[Data, DatasetInfo]:
    """Load the directed Telegram graph via torch_geometric_signed_directed.

    Telegram ships its own ``(N, 10)`` train/val/test masks, so it slots into
    the ``pyg_mask`` split path directly.
    """
    from torch_geometric_signed_directed import Telegram  # noqa: PLC0415

    from exp.data import DatasetInfo  # noqa: PLC0415
    from exp.datasets.assets import ensure_tarball  # noqa: PLC0415

    # Prefer the raw files mirrored on the dataset release; fall back to the
    # torch_geometric_signed_directed downloader when the mirror is absent.
    if not os.path.isdir(os.path.join(root, "telegram")):
        ensure_tarball("pyg_raw__telegram.tar.gz", root)
    ds = Telegram(os.path.join(root, "telegram"))
    data = cast(Data, ds[0])
    assert data.x is not None and isinstance(data.y, torch.Tensor)

    info = DatasetInfo(
        name="telegram",
        num_features=int(data.x.size(1)),
        num_classes=int(data.y.max().item()) + 1,
        num_splits=int(data.train_mask.size(1)) if data.train_mask.dim() > 1 else 1,
        metric="acc",
        split_type="pyg_mask",
    )
    return data, info


__all__ = [
    "available_directed",
    "directed_dir",
    "is_directed_citation",
    "is_directed_npz",
    "load_directed_citation",
    "load_directed_npz",
    "load_telegram",
]
