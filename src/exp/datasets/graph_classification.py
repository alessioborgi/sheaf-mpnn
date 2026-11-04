# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""TUDataset graph-classification DataModules.

This module provides specialized Lightning DataModules for processing TUDatasets
(like MUTAG, PROTEINS, COLLAB) in both homogeneous and heterogeneous contexts.

Class hierarchy
---------------
``GCSheafDataModule``     - homogeneous TU datasets (all types forced to 0 via
                            :class:`SetHomogeneous`).
``GCHetSheafDataModule``  - heterogeneous TU datasets (real node/edge types
                            extracted from one-hot columns via :class:`ExtractTypes`).

Both inherit from :class:`~exp.data.SheafDataModule`, re-using its ``__init__``
signature ``(name, root, fold)`` and the ``info`` property.  They override
``setup()`` (TUDataset loading + stratified split) and all three dataloader
methods (batched graph DataLoaders instead of single full-graph loaders).

Legacy ``GraphClassificationDataModule`` is kept for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from exp.data import SheafDataModule
from exp.datasets.info import GCDatasetInfo

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class ExtractTypes:
    """Extract node/edge types from one-hot feature columns.

    Many TUDatasets encode discrete node/edge labels as one-hot prefixes in the
    feature tensors. This transform extracts those labels into dedicated
    `node_type` and `edge_type` tensors, which sheaf models use to parameterize
    stalk and restriction maps.

    Args:
        node_type_cols: ``[start, end]`` range of one-hot node-type columns in `data.x`.
        edge_type_cols: ``[start, end]`` range of one-hot edge-type columns in
            `data.edge_attr`.
        node_feature_type: Strategy for nodes without features:
            - ``"constant"``: Fill with a constant value (1.0).
            - ``"degree"``: Use the node degree as a feature.
    """

    def __init__(
        self,
        node_type_cols: list[int] | None = None,
        edge_type_cols: list[int] | None = None,
        node_feature_type: Literal["constant", "degree"] = "constant",
    ):
        self.node_type_cols = node_type_cols
        self.edge_type_cols = edge_type_cols
        self.node_feature_type = node_feature_type

    def __call__(self, data: Data) -> Data:
        """Apply the type extraction to a single Data object."""
        assert data.num_nodes is not None
        # Handle cases where input data has no features
        if data.x is None:
            if self.node_feature_type == "degree":
                assert data.edge_index is not None
                deg = degree(
                    data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float
                )
                data.x = deg.view(-1, 1)
            else:
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)

        # 1. Extract Node Types
        node_type = None
        if self.node_type_cols is not None and data.x is not None:
            start, end = self.node_type_cols
            if data.x.size(1) >= end:
                # Argmax over the one-hot range identifies the type index
                node_type = data.x[:, start:end].argmax(dim=1)

                # Remove the extracted columns from the features
                if start == 0 and end >= data.x.size(1):
                    # Features were entirely one-hot types; replace with fallback
                    if self.node_feature_type == "degree":
                        assert data.edge_index is not None
                        deg = degree(
                            data.edge_index[0],
                            num_nodes=data.x.size(0),
                            dtype=data.x.dtype,
                        )
                        data.x = deg.view(-1, 1)
                    else:
                        data.x = torch.ones(data.x.size(0), 1, dtype=data.x.dtype)
                elif start == 0:
                    data.x = data.x[:, end:]
                elif end >= data.x.size(1):
                    data.x = data.x[:, :start]
                else:
                    data.x = torch.cat([data.x[:, :start], data.x[:, end:]], dim=1)
            else:
                node_type = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            node_type = torch.zeros(data.num_nodes, dtype=torch.long)

        # 2. Extract Edge Types
        edge_type = None
        if (
            self.edge_type_cols is not None
            and hasattr(data, "edge_attr")
            and data.edge_attr is not None
        ):
            start, end = self.edge_type_cols
            if data.edge_attr.dim() == 1:
                # Edge attr is already categorical indices
                edge_type = data.edge_attr.long()
            elif data.edge_attr.dim() == 2 and data.edge_attr.size(1) >= end:
                # Edge attr is one-hot; extract and remove
                edge_type = data.edge_attr[:, start:end].argmax(dim=1)
                if start == 0 and end >= data.edge_attr.size(1):
                    # Replace with dummy constant weight if all attrs were types
                    data.edge_attr = torch.ones(
                        data.edge_attr.size(0), 1, dtype=data.edge_attr.dtype
                    )
                elif start == 0:
                    data.edge_attr = data.edge_attr[:, end:]
                elif end >= data.edge_attr.size(1):
                    data.edge_attr = data.edge_attr[:, :start]
                else:
                    data.edge_attr = torch.cat(
                        [data.edge_attr[:, :start], data.edge_attr[:, end:]], dim=1
                    )
        else:
            # Fallback to homogeneous edge types
            if hasattr(data, "edge_index") and data.edge_index is not None:
                num_edges = data.edge_index.size(1)
                edge_type = torch.zeros(num_edges, dtype=torch.long)

        data.node_type = node_type
        data.edge_type = edge_type
        # Add batch vector for pooling operations (default to 0 for single graph)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    def __repr__(self) -> str:
        """Return transform description."""
        return (
            f"ExtractTypes(node_type_cols={self.node_type_cols},"
            f" edge_type_cols={self.edge_type_cols})"
        )


class SetHomogeneous:
    """Force homogeneous mode: set all node/edge types to 0.

    This transform ensures that even if the dataset has multiple labels, the
    model treats all nodes and edges as having the same type. This is the
    standard GNN approach.
    """

    def __init__(self, node_feature_type: Literal["constant", "degree"] = "constant"):
        self.node_feature_type = node_feature_type

    def __call__(self, data: Data) -> Data:
        assert data.num_nodes is not None
        if data.x is None:
            if self.node_feature_type == "degree":
                assert data.edge_index is not None
                deg = degree(
                    data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float
                )
                data.x = deg.view(-1, 1)
            else:
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)

        data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)

        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
        else:
            data.edge_type = None

        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    def __repr__(self) -> str:
        """Return transform description."""
        return "SetHomogeneous()"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _compute_type_counts(dataset: Dataset) -> tuple[int, int]:
    """Scan all graphs in the dataset and return the total number of node/edge types.

    This is used to populate the metadata needed to initialize sheaf restriction maps.
    """
    max_nt = 0
    max_et = 0
    for data in dataset:
        if hasattr(data, "node_type") and data.node_type is not None:
            max_nt = max(max_nt, int(data.node_type.max().item()) + 1)
        if (
            hasattr(data, "edge_type")
            and data.edge_type is not None
            and data.edge_type.numel() > 0
        ):
            max_et = max(max_et, int(data.edge_type.max().item()) + 1)
    return max(max_nt, 1), max(max_et, 1)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GCDatasetEntry:
    tu_name: str
    metric: str = "acc"


gc_homo_registry: dict[str, GCDatasetEntry] = {
    "collab": GCDatasetEntry("COLLAB"),
    "imdb_b": GCDatasetEntry("IMDB-B"),
    "imdb_m": GCDatasetEntry("IMDB-M"),
    "reddit_b": GCDatasetEntry("REDDIT-B"),
    "reddit_5k": GCDatasetEntry("REDDIT-MULTI-5K"),
}


@dataclass(frozen=True)
class GCHetDatasetEntry:
    tu_name: str
    node_type_cols: list[int]
    edge_type_cols: list[int] | None = None
    node_feature_type: Literal["constant", "degree"] = "constant"
    metric: str = "acc"


gc_het_registry: dict[str, GCHetDatasetEntry] = {
    "mutag": GCHetDatasetEntry("MUTAG", [0, 7], [0, 4]),
    "proteins": GCHetDatasetEntry("PROTEINS", [0, 3], None),
    "enzymes": GCHetDatasetEntry("ENZYMES", [0, 3], None),
    "nci1": GCHetDatasetEntry("NCI1", [0, 37], None),
    "nci109": GCHetDatasetEntry("NCI109", [0, 38], None),
    "ptcmr": GCHetDatasetEntry("PTCMR", [0, 18], [0, 4]),
    "dd": GCHetDatasetEntry("DD", [0, 89], None),
}


# ---------------------------------------------------------------------------
# GCSheafDataModule  homogeneous graph classification
# ---------------------------------------------------------------------------


class GCSheafDataModule(SheafDataModule):
    """Homogeneous TUDataset DataModule for graph classification.

    All nodes and edges are treated as having a single type (forced to 0).
    It implements a randomized proportional split by default, using the `fold`
    parameter to ensure different reproducible splits for cross-validation.

    Args:
        name: Name of the dataset (e.g., "COLLAB").
        root: Directory to store the raw/processed data.
        fold: Fold index [0-9] for cross-validation splitting.
        batch_size: Number of graphs per batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory for GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
        train_split: Fraction of dataset for training (default 0.8).
        val_split: Fraction of dataset for validation (default 0.1).
    """

    def __init__(
        self,
        name: str,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            name=name,
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._train_split = train_split
        self._val_split = val_split

    def _build_transform(self) -> SetHomogeneous | ExtractTypes:
        return SetHomogeneous()

    def _get_tu_name(self) -> str:
        entry = gc_homo_registry[self._name.lower()]
        return entry.tu_name

    def _get_metric(self) -> str:
        entry = gc_homo_registry[self._name.lower()]
        return entry.metric

    def setup(self, stage: str | None = None) -> None:
        """Load dataset and create train/val/test splits."""
        if self._train_dataset is not None:
            return  # idempotent

        transform = self._build_transform()
        tu_name = self._get_tu_name()
        metric = self._get_metric()

        full = TUDataset(self._root, tu_name, transform=transform)

        # Sequential randomized splitting
        n = len(full)
        n_train = int(n * self._train_split)
        n_val = int(n * self._val_split)
        # Use fold to drive the seed for reproducible cross-validation
        gen = torch.Generator().manual_seed(42 + self._fold)
        perm = torch.randperm(n, generator=gen)

        self._train_dataset = cast(Dataset, full[perm[:n_train]])
        self._val_dataset = cast(Dataset, full[perm[n_train : n_train + n_val]])
        self._test_dataset = cast(Dataset, full[perm[n_train + n_val :]])

        num_node_types, num_edge_types = _compute_type_counts(full)
        num_classes = int(full.num_classes)

        # Populate metadata for the model
        self._info = GCDatasetInfo(
            name=self._name.lower(),
            num_features=int(full.num_node_features),
            num_classes=num_classes,
            num_splits=10,
            metric=metric,
            split_type="seed",
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            task="binary" if num_classes == 2 else "multiclass",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        """Return counts of graphs in each split."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing .split_sizes")
        assert self._val_dataset is not None
        assert self._test_dataset is not None
        return (
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
        )

    # num_nodes / num_edges / homophily are not meaningful for collection of graphs.
    @property
    def num_nodes(self) -> int:  # type: ignore[override]
        raise AttributeError(
            "num_nodes is not defined for graph-classification DataModules"
        )

    @property
    def num_edges(self) -> int:  # type: ignore[override]
        raise AttributeError(
            "num_edges is not defined for graph-classification DataModules"
        )

    @property
    def homophily(self) -> float:  # type: ignore[override]
        raise AttributeError(
            "homophily is not defined for graph-classification DataModules"
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Shuffled mini-batch loader over the training graphs."""
        assert self._train_dataset is not None, "Call setup() first"
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers and self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Mini-batch loader over the validation graphs."""
        assert self._val_dataset is not None, "Call setup() first"
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers and self._num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Mini-batch loader over the test graphs."""
        assert self._test_dataset is not None, "Call setup() first"
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers and self._num_workers > 0,
        )


# ---------------------------------------------------------------------------
# GCHetSheafDataModule  heterogeneous graph classification
# ---------------------------------------------------------------------------


class GCHetSheafDataModule(GCSheafDataModule):
    """Heterogeneous TUDataset DataModule for graph classification.

    This DataModule extracts discrete node and edge types from the raw dataset,
    allowing the sheaf model to assign different stalk and restriction map
    parameters to different types of entities.
    """

    def _build_transform(self) -> ExtractTypes:
        entry = gc_het_registry[self._name.lower()]
        return ExtractTypes(
            node_type_cols=entry.node_type_cols,
            edge_type_cols=entry.edge_type_cols,
            node_feature_type=entry.node_feature_type,
        )

    def _get_tu_name(self) -> str:
        return gc_het_registry[self._name.lower()].tu_name

    def _get_metric(self) -> str:
        return gc_het_registry[self._name.lower()].metric

    @property
    def num_node_types(self) -> int:
        """Total number of node types found in the dataset."""
        if self._info is None:
            raise RuntimeError("Call setup() before accessing .num_node_types")
        assert isinstance(self._info, GCDatasetInfo)
        return self._info.num_node_types

    @property
    def num_edge_types(self) -> int:
        """Total number of edge types found in the dataset."""
        if self._info is None:
            raise RuntimeError("Call setup() before accessing .num_edge_types")
        assert isinstance(self._info, GCDatasetInfo)
        return self._info.num_edge_types


# ---------------------------------------------------------------------------
# Legacy GraphClassificationDataModule (TMP-compatible)
# ---------------------------------------------------------------------------


class GraphClassificationDataModule(GCSheafDataModule):
    """Legacy DataModule matching the TMP/datasets API.

    Maintained to ensure that existing sweep scripts and configurations
    remain functional while the codebase transitions to the SheafDataModule hierarchy.
    """

    def __init__(
        self,
        root: str,
        name: str,
        homogeneous: bool = False,
        node_type_cols: list[int] | None = None,
        edge_type_cols: list[int] | None = None,
        node_feature_type: Literal["constant", "degree"] = "constant",
        num_workers: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
        batch_size: int = 1,
        **kwargs: Any,
    ):
        # Derive fold from seed for interface compatibility
        fold = max(0, seed - 42)
        super().__init__(name=name, root=root, fold=fold, batch_size=batch_size)
        self._homogeneous = homogeneous
        self._node_type_cols = node_type_cols
        self._edge_type_cols = edge_type_cols
        self._node_feature_type = node_feature_type
        self._num_workers = num_workers
        self._train_split = train_split
        self._val_split = val_split
        self._seed = seed

    def _build_transform(self):  # type: ignore[override]
        if self._homogeneous:
            return SetHomogeneous(node_feature_type=self._node_feature_type)
        return ExtractTypes(
            node_type_cols=self._node_type_cols,
            edge_type_cols=self._edge_type_cols,
            node_feature_type=self._node_feature_type,
        )

    def _get_tu_name(self) -> str:
        return self._name

    def _get_metric(self) -> str:
        return "acc"

    def setup(self, stage: str | None = None) -> None:
        """Load the TU dataset and split it with a seeded random permutation."""
        if self._train_dataset is not None:
            return

        transform = self._build_transform()
        tu_name = self._get_tu_name()

        full = TUDataset(self._root, tu_name, transform=transform)

        n = len(full)
        n_train = int(n * self._train_split)
        n_val = int(n * self._val_split)
        gen = torch.Generator().manual_seed(self._seed)
        perm = torch.randperm(n, generator=gen)

        self._train_dataset = cast(Dataset, full[perm[:n_train]])
        self._val_dataset = cast(Dataset, full[perm[n_train : n_train + n_val]])
        self._test_dataset = cast(Dataset, full[perm[n_train + n_val :]])

        num_node_types, num_edge_types = _compute_type_counts(full)
        num_classes = int(full.num_classes)

        self._info = GCDatasetInfo(
            name=self._name.lower(),
            num_features=int(full.num_node_features),
            num_classes=num_classes,
            num_splits=10,
            metric="acc",
            split_type="seed",
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            task="binary" if num_classes == 2 else "multiclass",
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Shuffled mini-batch loader over the training graphs."""
        assert self._train_dataset is not None, "Call setup() first"
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers and self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Mini-batch loader over the validation graphs."""
        assert self._val_dataset is not None, "Call setup() first"
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        """Mini-batch loader over the test graphs."""
        assert self._test_dataset is not None, "Call setup() first"
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._num_workers > 0,
        )


# ---------------------------------------------------------------------------
# Concrete shims
# ---------------------------------------------------------------------------


class MUTAGDataModule(GCHetSheafDataModule):
    """MUTAG: 188 mutagenic compounds (7 atom types, 4 bond types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "mutag",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class PROTEINSDataModule(GCHetSheafDataModule):
    """PROTEINS: 1113 proteins (3 node types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "proteins",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class ENZYMESDataModule(GCHetSheafDataModule):
    """ENZYMES: 600 enzymes, 6 classes (3 node types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "enzymes",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class NCI1DataModule(GCHetSheafDataModule):
    """NCI1: 4110 compounds (37 atom types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "nci1",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class NCI109DataModule(GCHetSheafDataModule):
    """NCI109: Chemical compounds for cancer screening (38 atom types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "nci109",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class PTCMRDataModule(GCHetSheafDataModule):
    """PTC_MR: Predictive Toxicology Challenge (18 atom types, 4 bond types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "ptcmr",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class DDDataModule(GCHetSheafDataModule):
    """DD: Protein structures, large graphs (89 node types)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "dd",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class COLLABDataModule(GCSheafDataModule):
    """COLLAB: Scientific collaborations (homogeneous)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "collab",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class IMDBBinaryDataModule(GCSheafDataModule):
    """IMDB-BINARY: Movie collaborations (homogeneous)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "imdb_b",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class IMDBMultiDataModule(GCSheafDataModule):
    """IMDB-MULTI: Movie collaborations, 3 classes (homogeneous)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "imdb_m",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class REDDITBinaryDataModule(GCSheafDataModule):
    """REDDIT-BINARY: Reddit threads (homogeneous)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "reddit_b",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )


class REDDIT5KDataModule(GCSheafDataModule):
    """REDDIT-MULTI-5K: Reddit threads, 5 classes (homogeneous)."""

    def __init__(
        self,
        root: str = "data",
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(
            "reddit_5k",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
        )
