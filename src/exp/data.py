# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Dataset loading and Lightning DataModule for the NSD benchmark suite.

Supported datasets:

- Homophilic: cora, citeseer, pubmed
- Heterophilic: chameleon, squirrel, chameleon_filtered, squirrel_filtered,
  cornell, texas, wisconsin, film
- Heterophilous (Platonov et al. 2023): amazon_ratings, minesweeper, questions,
  roman_empire, tolokers
- OGB (Hu et al. 2020): ogbn_arxiv, ogbn_products (official single split,
  accuracy); ogbn_proteins (official single split, mean ROC-AUC over the 112
  binary protein-function targets)
- LINKX (Lim et al. 2021): pokec (5 official 50/25/25 splits over labeled
  nodes, accuracy)

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
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

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
    # "multilabel" keeps y as an [N, C] binary matrix trained with BCE;
    # "multiclass" is the default [N] integer-label cross-entropy task.
    task: str = "multiclass"


# Datasets evaluated with ROC-AUC (binary or multilabel node classification).
ROC_AUC_DATASETS: frozenset[str] = frozenset(
    {"minesweeper", "tolokers", "questions", "ogbn_proteins"}
)

# Multilabel datasets: y is an [N, C] binary matrix, one column per target.
MULTILABEL_DATASETS: frozenset[str] = frozenset({"ogbn_proteins"})

# Canonical dataset name (loader_kind, pyg_key).
_LOADER: dict[str, tuple[str, str]] = {
    "cora": ("planetoid", "Cora"),
    "citeseer": ("planetoid", "CiteSeer"),
    "pubmed": ("planetoid", "PubMed"),
    "chameleon": ("wiki", "chameleon"),
    "squirrel": ("wiki", "squirrel"),
    "chameleon_filtered": ("filtered_wiki", "chameleon_filtered"),
    "squirrel_filtered": ("filtered_wiki", "squirrel_filtered"),
    "cornell": ("webkb", "Cornell"),
    "texas": ("webkb", "Texas"),
    "wisconsin": ("webkb", "Wisconsin"),
    "film": ("actor", ""),
    "amazon_ratings": ("heterophilous", "amazon-ratings"),
    "minesweeper": ("heterophilous", "minesweeper"),
    "questions": ("heterophilous", "questions"),
    "roman_empire": ("heterophilous", "roman-empire"),
    "tolokers": ("heterophilous", "tolokers"),
    "ogbn_arxiv": ("ogb", "ogbn-arxiv"),
    "ogbn_products": ("ogb", "ogbn-products"),
    "ogbn_proteins": ("ogb", "ogbn-proteins"),
    "pokec": ("linkx", "pokec"),
}

# "fil" accepted as an alias for "film".
_ALIASES: dict[str, str] = {"fil": "film"}

# Datasets whose splits live in exp/splits/*.npz (48/32/20, 10 folds).
NPZ_SPLIT_DATASETS: frozenset[str] = frozenset(
    {
        "cora",
        "citeseer",
        "pubmed",
        "chameleon",
        "squirrel",
        "cornell",
        "texas",
        "wisconsin",
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


def load_dataset(
    name: str, root: str = "exp/data", to_undirected: bool = True
) -> tuple[Data, DatasetInfo]:
    """Load a benchmark dataset, downloading it automatically if needed.

    Args:
        name: Dataset identifier.  One of the supported names (standard,
            synthetic, or directed).
        root: Base directory used by PyG for download caching.
        to_undirected: If True (the default), symmetrise the graph after
            loading via :func:`~torch_geometric.utils.to_undirected` -- the
            sheaf Laplacian assumes a symmetric graph, so callers must opt in
            to directionality.  Pass False to keep the original directed edges
            (the ``--directed`` flag); no-op for already-undirected datasets.

    Returns:
        data: A single ``torch_geometric.data.Data`` object.
        info: A :class:`DatasetInfo` with metadata about the dataset.
    """
    # Stock PyG loaders (geom-gcn / planetoid) symmetrise inside the impl only
    # when an undirected graph is requested; otherwise they keep DSNN's directed
    # preprocessing (remove_self_loops + coalesce, no to_undirected).
    data, info = _load_dataset_impl(name, root, symmetrize=to_undirected)

    if to_undirected and not _is_undirected(data):
        # Local alias: the bool parameter shadows the module-level import.
        from torch_geometric.utils import (
            to_undirected as _to_undirected,  # noqa: PLC0415
        )

        data = data.clone()
        assert data.edge_index is not None
        data.edge_index = _to_undirected(data.edge_index, num_nodes=data.num_nodes)

    return data, info


def _is_undirected(data: Data) -> bool:
    from torch_geometric.utils import is_undirected as _iu  # noqa: PLC0415

    assert data.edge_index is not None
    return bool(_iu(data.edge_index, num_nodes=data.num_nodes))


# Stock PyG datasets whose upstream raw files are mirrored on the dataset
# release (their canonical hosts are research repos that may disappear or
# change bytes; the mirror pins the data our results were produced with).
_PYG_RAW_MIRROR: frozenset[str] = frozenset(
    {"cornell", "texas", "wisconsin", "chameleon", "squirrel", "film"}
)


def _prefetch_pyg_raw(name: str, root: str) -> None:
    """Pre-populate a stock PyG dataset's cache from the dataset release.

    ``pyg_raw__<name>.tar.gz`` holds the dataset's ``raw`` subtree exactly as
    the PyG loader caches it (``processed/`` is always rebuilt locally).  When
    the mirror is unavailable this is a no-op and the PyG loader downloads
    from its upstream source as usual.
    """
    if name not in _PYG_RAW_MIRROR or os.path.isdir(os.path.join(root, name)):
        return
    from exp.datasets.assets import ensure_tarball  # noqa: PLC0415

    ensure_tarball(f"pyg_raw__{name}.tar.gz", root)


def _load_dataset_impl(
    name: str, root: str = "exp/data", symmetrize: bool = True
) -> tuple[Data, DatasetInfo]:
    """Dispatch a dataset name to its loader (no post-processing).

    Directed sources (DSNN) take precedence over the standard PyG loaders for
    overlapping names: ``cora``/``citeseer``/``pubmed`` load the directed
    citation graphs, and the Platonov heterophilous graphs
    (``roman_empire``, ``amazon_ratings``, ``questions``, ``chameleon_filtered``,
    ``squirrel_filtered``, ``minesweeper``, ``tolokers``) load their directed
    ``.npz`` variants.  Symmetrise them via the ``to_undirected`` option in
    :func:`load_dataset` to recover the undirected graph.
    """
    name = _canonical(name)

    # Synthetic directed (DSBM) graphs are loaded from pickled Data objects and
    # processed DSNN-style; the loader downloads the ``.pk`` file on first use.
    from exp.datasets.synthetic import is_synthetic, load_synthetic  # noqa: PLC0415

    if is_synthetic(name, root) or name.startswith("dataset_nodes"):
        return load_synthetic(name, root=root)

    # Real-world directed graphs shipped by DSNN.  Matched by their on-disk name:
    # Platonov-style .npz, directed citation pickles, or Telegram.
    from exp.datasets.directed import (  # noqa: PLC0415
        is_directed_citation,
        is_directed_npz,
        load_directed_citation,
        load_directed_npz,
        load_telegram,
    )

    if is_directed_npz(name, root):
        return load_directed_npz(name, root=root)
    if is_directed_citation(name, root):
        return load_directed_citation(name, root=root)
    if name == "telegram":
        return load_telegram(root=root)

    if name not in _LOADER:
        raise ValueError(
            f"Unknown dataset {name!r}.  Supported: {sorted(_LOADER.keys())}"
        )

    _prefetch_pyg_raw(name, root)

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

    elif kind == "linkx":
        from exp.datasets.linkx import load_linkx  # noqa: PLC0415

        data = load_linkx(name, ds_root)
        num_splits = int(data.train_mask.size(1))

    elif kind == "ogb":
        # OGB manages its own download/caching under ds_root. The official
        # single split from get_idx_split() is stamped as 1-D boolean masks,
        # so apply_split's pyg_mask path returns it unchanged for every fold.
        from ogb.nodeproppred import PygNodePropPredDataset  # noqa: PLC0415
        from torch_geometric.data.data import (  # noqa: PLC0415
            DataEdgeAttr,
            DataTensorAttr,
        )
        from torch_geometric.data.storage import GlobalStorage  # noqa: PLC0415

        # ogb<=1.3.6 calls torch.load on its processed cache without
        # weights_only=False; torch>=2.6 defaults to weights_only=True, which
        # rejects PyG's storage classes unless they are allow-listed.
        torch.serialization.add_safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage]
        )
        ds = PygNodePropPredDataset(name=pykey, root=ds_root)
        data = cast(Data, ds[0])
        if data.x is None:
            # ogbn-proteins ships no node features; the standard full-batch
            # input is the mean of each node's incident edge features.
            assert data.edge_attr is not None and data.edge_index is not None
            n = int(data.num_nodes or 0)
            src = data.edge_index[0]
            deg = torch.zeros(n).index_add_(0, src, torch.ones(src.size(0)))
            feat = torch.zeros(n, data.edge_attr.size(1))
            feat.index_add_(0, src, data.edge_attr.float())
            data.x = feat / deg.clamp(min=1).unsqueeze(-1)
            data.edge_attr = None
        assert data.x is not None
        split_idx = ds.get_idx_split()
        n = int(data.x.size(0))
        for split_name, mask_name in (
            ("train", "train_mask"),
            ("valid", "val_mask"),
            ("test", "test_mask"),
        ):
            mask = torch.zeros(n, dtype=torch.bool)
            mask[split_idx[split_name]] = True
            data[mask_name] = mask
        assert isinstance(data.y, torch.Tensor)
        if name in MULTILABEL_DATASETS:
            # Binary target matrix stays [N, C] and float for the BCE loss.
            data.y = data.y.float()
        else:
            # OGB ships labels as [N, 1]; the loss and metrics expect [N].
            data.y = data.y.view(-1).long()
        num_splits = 1

    else:
        raise AssertionError(f"Unhandled loader kind: {kind!r}")

    assert data.x is not None
    assert data.edge_index is not None
    # Stock PyG loaders ship raw directed edge lists.  Match DSNN's
    # preprocessing (utils/heterophilic.py, utils/classic.py): drop self-loops
    # and coalesce, keeping the graph directed; symmetrise only when an
    # undirected graph is requested (the sheaf Laplacian's default).
    edge_index, _ = remove_self_loops(data.edge_index)
    if symmetrize:
        edge_index = to_undirected(edge_index)
    data.edge_index = coalesce(edge_index, num_nodes=data.x.size(0))
    # Row-normalize features only where the reference protocol does (Bodnar
    # et al. on the Pei/planetoid sets); Platonov et al. use raw features.
    x = data.x
    if kind in ("planetoid", "wiki", "webkb", "actor"):
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1)
        data.x = x
    if kind == "planetoid":
        # Reference non_valid_samples: citeseer's isolated nodes carry all-zero
        # features and a fabricated label; splits.py drops them from all masks.
        data.non_valid_mask = x.sum(dim=-1) == 0
    assert isinstance(data.y, torch.Tensor)
    num_features = int(x.size(1))
    task = "multilabel" if name in MULTILABEL_DATASETS else "multiclass"
    if task == "multilabel":
        # One binary column per target; unlabeled (-1) entries cannot occur.
        num_classes = int(data.y.size(1))
    else:
        num_classes = int(data.y.max().item()) + 1
    split_type = "npz_file" if name in NPZ_SPLIT_DATASETS else "pyg_mask"

    info = DatasetInfo(
        name=name,
        num_features=num_features,
        num_classes=num_classes,
        num_splits=num_splits,
        metric=metric,
        split_type=split_type,
        task=task,
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
            to_undirected: bool = True,
        ) -> None:
            super().__init__()
            self._name = name
            self._root = root
            self._fold = fold
            self._to_undirected = to_undirected
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
        def num_edges_total(self) -> int:
            """Number of directed edges in ``edge_index`` (i.e. its column count)."""
            if self._data is None:
                raise RuntimeError("Call setup() before accessing .num_edges_total")
            return int(self._data.num_edges)

        @property
        def is_directed(self) -> bool:
            """True if the loaded graph is directed (edge_index is asymmetric)."""
            from torch_geometric.utils import is_undirected as _iu  # noqa: PLC0415

            if self._data is None:
                raise RuntimeError("Call setup() before accessing .is_directed")
            assert self._data.edge_index is not None
            return not bool(_iu(self._data.edge_index, num_nodes=self._data.num_nodes))

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
            """Load the dataset once and apply the split for the configured fold."""
            if self._data is None:
                self._data, self._info = load_dataset(
                    self._name, root=self._root, to_undirected=self._to_undirected
                )
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
            """Full-batch loader over the split graph; training uses ``train_mask``."""
            return self._loader()

        def val_dataloader(self) -> _PyGLoader:
            """Full-batch loader over the split graph; validation uses ``val_mask``."""
            return self._loader()

        def test_dataloader(self) -> _PyGLoader:
            """Full-batch loader over the split graph; testing uses ``test_mask``."""
            return self._loader()

except ImportError:  # pragma: no cover
    pass  # lightning is optional; SheafDataModule is unavailable without it
