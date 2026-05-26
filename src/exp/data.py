# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Dataset loading and Lightning DataModule for the NSD benchmark suite.

Supported datasets
------------------
Homophilic  : cora, citeseer
Heterophilic: chameleon, squirrel, chameleon_filtered, squirrel_filtered,
            cornell, texas, film
Heterophilous (Platonov et al. 2023):
            amazon_ratings, minesweeper, questions, roman_empire, tolokers

All datasets are downloaded automatically to ``root`` on first use.
Filtered Wikipedia datasets are fetched from the yandex-research GitHub
release if not already cached locally.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import (
    Actor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.utils import coalesce, to_undirected

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    """Lightweight descriptor returned alongside a loaded Data object."""

    name: str
    num_features: int
    num_classes: int
    num_splits: int  # number of available train/val/test folds
    metric: str  # "acc" | "roc_auc" -- depends on dataset evaluation protocol
    split_type: str  # "npz_file" | "pyg_mask"


# Datasets evaluated with ROC-AUC (binary node classification).
ROC_AUC_DATASETS: frozenset[str] = frozenset({"minesweeper", "tolokers", "questions"})

# Canonical dataset name (loader_kind, pyg_key).
_LOADER: dict[str, tuple[str, str]] = {
    "cora": ("planetoid", "Cora"),
    "citeseer": ("planetoid", "CiteSeer"),
    "chameleon": ("wiki", "chameleon"),
    "squirrel": ("wiki", "squirrel"),
    "chameleon_filtered": ("filtered_wiki", "chameleon_filtered"),
    "squirrel_filtered": ("filtered_wiki", "squirrel_filtered"),
    "cornell": ("webkb", "Cornell"),
    "texas": ("webkb", "Texas"),
    "film": ("actor", ""),
    "amazon_ratings": ("heterophilous", "amazon-ratings"),
    "minesweeper": ("heterophilous", "minesweeper"),
    "questions": ("heterophilous", "questions"),
    "roman_empire": ("heterophilous", "roman-empire"),
    "tolokers": ("heterophilous", "tolokers"),
}

# "fil" accepted as an alias for "film".
_ALIASES: dict[str, str] = {"fil": "film"}

# Datasets whose splits live in exp/splits/*.npz (60/20/20, 10 folds).
NPZ_SPLIT_DATASETS: frozenset[str] = frozenset(
    {
        "cora",
        "citeseer",
        "chameleon",
        "squirrel",
        "cornell",
        "texas",
        "film",
    }
)


def _canonical(name: str) -> str:
    name = name.lower().replace("-", "_").strip()
    return _ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Custom dataset for Filtered Wikipedia networks (Platonov et al., 2023).
# ---------------------------------------------------------------------------

_FILTERED_WIKI_URLS: dict[str, str] = {
    "chameleon_filtered": (
        "https://github.com/yandex-research/heterophilous-graphs"
        "/raw/main/data/chameleon_filtered.npz"
    ),
    "squirrel_filtered": (
        "https://github.com/yandex-research/heterophilous-graphs"
        "/raw/main/data/squirrel_filtered.npz"
    ),
}


class FilteredWikipediaDataset(InMemoryDataset):
    """Filtered chameleon / squirrel networks (Platonov et al., 2023).

    The raw ``.npz`` file embeds 10 pre-computed train/val/test splits as
    boolean mask matrices of shape ``(10, N)``.  After processing they are
    stored as ``(N, 10)`` tensors so they match the convention used by other
    PyG heterophilous datasets.
    """

    def __init__(self, root: str, name: str) -> None:
        assert name in _FILTERED_WIKI_URLS, f"Unknown filtered dataset: {name!r}"
        self._ds_name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self._ds_name}.npz"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self._ds_name}.pt"]

    def download(self) -> None:
        url = _FILTERED_WIKI_URLS[self._ds_name]
        dst = os.path.join(self.raw_dir, f"{self._ds_name}.npz")
        print(f"Downloading {self._ds_name} from\n  {url}")
        urllib.request.urlretrieve(url, dst)

    def process(self) -> None:
        npz = np.load(os.path.join(self.raw_dir, f"{self._ds_name}.npz"))

        x = torch.from_numpy(npz["node_features"]).float()
        y = torch.from_numpy(npz["node_labels"]).long()

        edges = torch.from_numpy(npz["edges"]).t().contiguous()  # (2, E)
        edge_index = to_undirected(edges)
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        # raw masks: (10, N) bool  ->  (N, 10) for consistency with PyG convention
        train_mask = torch.from_numpy(npz["train_masks"]).T.bool()
        val_mask = torch.from_numpy(npz["val_masks"]).T.bool()
        test_mask = torch.from_numpy(npz["test_masks"]).T.bool()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        torch.save(self.collate([data]), self.processed_paths[0])


def load_dataset(name: str, root: str = "exp/data") -> tuple[Data, DatasetInfo]:
    """Load a benchmark dataset, downloading it automatically if needed.

    Args:
        name: Dataset identifier.  One of the 14 supported names.
        root: Base directory used by PyG for download caching.

    Returns:
        data: A single ``torch_geometric.data.Data`` object.
        info: A :class:`DatasetInfo` with metadata about the dataset.
    """
    name = _canonical(name)
    if name not in _LOADER:
        raise ValueError(
            f"Unknown dataset {name!r}.  Supported: {sorted(_LOADER.keys())}"
        )

    kind, pykey = _LOADER[name]
    metric = "roc_auc" if name in ROC_AUC_DATASETS else "acc"
    ds_root = os.path.join(root, name)

    if kind == "planetoid":
        ds = Planetoid(ds_root, pykey, split="public")
        data = cast(Data, ds[0])
        # Planetoid ships a single fixed split; 10-fold .npz files are used instead.
        num_splits = 10

    elif kind == "wiki":
        ds = WikipediaNetwork(ds_root, pykey, geom_gcn_preprocess=True)
        data = cast(Data, ds[0])
        num_splits = int(data.train_mask.size(1)) if data.train_mask.dim() > 1 else 1

    elif kind == "webkb":
        ds = WebKB(ds_root, pykey)
        data = cast(Data, ds[0])
        num_splits = 10  # 10-fold .npz files cover these datasets.

    elif kind == "actor":
        ds = Actor(ds_root)
        data = cast(Data, ds[0])
        num_splits = 10  # 10-fold .npz files cover film.

    elif kind == "heterophilous":
        ds = HeterophilousGraphDataset(ds_root, pykey)
        data = cast(Data, ds[0])
        num_splits = int(data.train_mask.size(1)) if data.train_mask.dim() > 1 else 10

    elif kind == "filtered_wiki":
        ds = FilteredWikipediaDataset(ds_root, name)
        data = cast(Data, ds[0])
        num_splits = int(data.train_mask.size(1)) if data.train_mask.dim() > 1 else 10

    else:
        raise AssertionError(f"Unhandled loader kind: {kind!r}")

    assert data.x is not None
    assert isinstance(data.y, torch.Tensor)
    num_features = int(data.x.size(1))
    num_classes = int(data.y.max().item()) + 1
    split_type = "npz_file" if name in NPZ_SPLIT_DATASETS else "pyg_mask"

    info = DatasetInfo(
        name=name,
        num_features=num_features,
        num_classes=num_classes,
        num_splits=num_splits,
        metric=metric,
        split_type=split_type,
    )
    return data, info


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

try:
    from lightning import LightningDataModule
    from torch_geometric.loader import DataLoader as _PyGLoader

    class SheafDataModule(LightningDataModule):
        """Full-graph DataModule for transductive node classification.

        All three dataloaders return the same graph; the model selects nodes
        via ``train_mask`` / ``val_mask`` / ``test_mask`` inside each step.
        """

        def __init__(
            self,
            name: str,
            root: str = "exp/data",
            fold: int = 0,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
        ) -> None:
            super().__init__()
            self._name = name
            self._root = root
            self._fold = fold
            self._batch_size = batch_size
            self._num_workers = num_workers
            self._pin_memory = pin_memory
            self._persistent_workers = persistent_workers
            self._data: Data | None = None
            self._info: DatasetInfo | None = None
            self._split: Data | None = None

        @property
        def info(self) -> DatasetInfo:
            if self._info is None:
                raise RuntimeError("Call setup() before accessing .info")
            return self._info

        @property
        def num_nodes(self) -> int:
            """Number of nodes in the loaded graph (requires setup() to have run)."""
            if self._data is None:
                raise RuntimeError("Call setup() before accessing .num_nodes")
            assert self._data.num_nodes is not None
            return int(self._data.num_nodes)

        @property
        def num_edges(self) -> int:
            """Number of undirected edges (requires setup() to have run)."""
            if self._data is None:
                raise RuntimeError("Call setup() before accessing .num_edges")
            return int(self._data.num_edges) // 2

        @property
        def homophily(self) -> float:
            """Edge homophily h in [0,1]: fraction of same-class edges."""
            from torch_geometric.utils import homophily as _h

            if self._data is None:
                raise RuntimeError("Call setup() before accessing .homophily")
            assert isinstance(self._data.y, torch.Tensor)
            assert self._data.edge_index is not None
            return float(_h(self._data.edge_index, self._data.y, method="edge"))

        @property
        def split_sizes(self) -> tuple[int, int, int]:
            """(train, val, test) node counts for the current fold."""
            if self._split is None:
                raise RuntimeError("Call setup() before accessing .split_sizes")
            return (
                int(self._split.train_mask.sum()),
                int(self._split.val_mask.sum()),
                int(self._split.test_mask.sum()),
            )

        def setup(self, stage: str | None = None) -> None:
            if self._data is None:
                self._data, self._info = load_dataset(self._name, root=self._root)
            # Lazy import avoids the circular dependency: splits.py -> data.py
            from exp.splits import apply_split  # noqa: PLC0415

            assert self._info is not None
            self._split = apply_split(self._data, self._info, self._fold)

        def _loader(self) -> _PyGLoader:
            assert self._split is not None, "Call setup() first"
            return _PyGLoader(
                [self._split],
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=self._persistent_workers and self._num_workers > 0,
            )

        def train_dataloader(self) -> _PyGLoader:
            return self._loader()

        def val_dataloader(self) -> _PyGLoader:
            return self._loader()

        def test_dataloader(self) -> _PyGLoader:
            return self._loader()

except ImportError:
    pass  # lightning is optional; SheafDataModule is unavailable without it
