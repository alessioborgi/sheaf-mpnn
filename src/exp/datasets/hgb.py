# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""HGB (Heterogeneous Graph Benchmark) datasets.

This module integrates HGB datasets (IMDB, DBLP, ACM, Freebase) into the
sheaf neural network pipeline.

Node classification
-------------------
``HetSheafDataModule`` is the canonical entry point. It converts heterogeneous
HGB graphs into a homogeneous ``Data`` object while preserving type information
as node/edge attributes. This allows the model to leverage the graph's
heterogeneity through type-specific parameters.

Concrete shims: ``IMDBDataModule``, ``DBLPDataModule``, ``ACMDataModule``,
``FreebaseDataModule``.

Link prediction
---------------
Provides DataModules for HGB link prediction tasks (Amazon, PubMed, YouTube),
including data downloading, parsing, and splitting.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, cast

import lightning as L
import torch
from torch_geometric import transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.lightning import LightningLinkData
from torch_geometric.datasets import HGBDataset

from exp.data import SheafDataModule
from exp.datasets.info import HetDatasetInfo
from exp.datasets.utils import RemoveSelfLoops, SubsetTrainSplit

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HetDatasetEntry:
    """Metadata entry for an HGB node-classification dataset."""

    hgb_name: Literal["IMDB", "DBLP", "ACM", "Freebase"]
    target: str
    task: Literal["multiclass", "multilabel"]
    num_classes: int
    metric: Literal["acc", "roc_auc"] = "acc"


het_dataset_registry: dict[str, HetDatasetEntry] = {
    "imdb": HetDatasetEntry("IMDB", "movie", "multilabel", 5),
    "dblp": HetDatasetEntry("DBLP", "author", "multiclass", 4),
    "acm": HetDatasetEntry("ACM", "paper", "multiclass", 3),
    "freebase": HetDatasetEntry("Freebase", "book", "multiclass", 7),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_hgb_dataset(name: str, root: str) -> tuple[Data, HetDatasetInfo]:
    """Load an HGB node-classification dataset and convert to homogeneous form.

    The conversion process:
    1. Loads the raw HeteroData from HGBDataset.
    2. Applies a sequence of transforms (undirected, normalization, etc.).
    3. Flattens the HeteroData into a single Data object using `to_homogeneous()`.
    4. Preserves `node_type` and `edge_type` indices for the sheaf model.

    Args:
        name: Internal name of the dataset (e.g., "dblp").
        root: Root directory for the dataset.

    Returns:
        A tuple of (processed Data, metadata Info).
    """
    entry = het_dataset_registry[name.lower()]

    # Standard HGB preprocessing pipeline
    transform = T.Compose(
        [
            T.Constant(node_types=None),  # Add constant features if missing
            T.ToUndirected(),
            SubsetTrainSplit(val_ratio=0.1),  # Create validation split from training
            RemoveSelfLoops(),  # Prune self-loops while keeping attrs aligned
            T.NormalizeFeatures(),
            T.RemoveDuplicatedEdges(),
        ]
    )

    dataset = HGBDataset(root=root, name=entry.hgb_name, transform=transform)
    hetero: HeteroData = cast(HeteroData, dataset[0]).coalesce()

    # Flatten the heterogeneous graph. PyG's to_homogeneous() automatically
    # creates .node_type and .edge_type tensors that map each node/edge to
    # its original type index.
    data: Data = hetero.to_homogeneous()

    # HGB datasets ship with fixed train/test masks on the target node type.
    # We ensure these masks are correctly preserved after the flattening.
    for mask in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, mask) or getattr(data, mask) is None:
            raise RuntimeError(
                f"HGB {entry.hgb_name!r} is missing '{mask}' after to_homogeneous()."
                " Check that the HGB dataset ships train/test splits."
            )

    # Bundle metadata for the training module
    info = HetDatasetInfo(
        name=name.lower(),
        num_features=int(data.num_features),
        num_classes=entry.num_classes,
        num_splits=1,
        metric=entry.metric,
        split_type="hgb_builtin",
        num_node_types=int(data.num_node_types),
        num_edge_types=int(data.num_edge_types),
        node_type_names=tuple(data._node_type_names),
        edge_type_names=tuple(str(e) for e in data._edge_type_names),
        task=entry.task,
    )
    return data, info


# ---------------------------------------------------------------------------
# HetSheafDataModule
# ---------------------------------------------------------------------------


class HetSheafDataModule(SheafDataModule):
    """Full-graph DataModule for HGB heterogeneous node classification.

    This class bridges HGB's heterogeneous structure with the transductive
    sheaf training loop. It leverages `to_homogeneous()` to allow the use
    of standard graph-level convolution operations while retaining access
    to type information via node/edge type tensors.
    """

    def setup(self, stage: str | None = None) -> None:
        """Load and prepare the dataset for training."""
        if self._data is None:
            self._data, self._info = _load_hgb_dataset(self._name, root=self._root)

        # HGB datasets provide official, fixed train/val/test masks to ensure
        # standardized benchmarking results across different models and papers.
        # Consequently, cross-validation is not applicable, and we use the
        # built-in split directly.
        self._split = self._data

    # ------------------------------------------------------------------
    # Extra properties for heterogeneous metadata
    # ------------------------------------------------------------------

    @property
    def num_node_types(self) -> int:
        """Total number of node types in the flattened graph."""
        if self._info is None:
            raise RuntimeError("Call setup() before accessing .num_node_types")
        assert isinstance(self._info, HetDatasetInfo)
        return self._info.num_node_types

    @property
    def num_edge_types(self) -> int:
        """Total number of edge types in the flattened graph."""
        if self._info is None:
            raise RuntimeError("Call setup() before accessing .num_edge_types")
        assert isinstance(self._info, HetDatasetInfo)
        return self._info.num_edge_types


# ---------------------------------------------------------------------------
# Concrete shims (node classification)
# ---------------------------------------------------------------------------


class IMDBDataModule(HetSheafDataModule):
    """IMDB: Movie dataset (Movie, Actor, Director types)."""

    def __init__(
        self,
        root: str = DATA_DIR,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            "imdb", root=root, fold=fold, batch_size=batch_size, num_workers=num_workers
        )

    def __str__(self) -> str:
        """Return dataset name."""
        return "IMDB"


class DBLPDataModule(HetSheafDataModule):
    """DBLP: Bibliographic dataset (Author, Paper, Term, Venue types)."""

    def __init__(
        self,
        root: str = DATA_DIR,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            "dblp", root=root, fold=fold, batch_size=batch_size, num_workers=num_workers
        )

    def __str__(self) -> str:
        """Return dataset name."""
        return "DBLP"


class ACMDataModule(HetSheafDataModule):
    """ACM: Bibliographic dataset (Paper, Author, Subject types)."""

    def __init__(
        self,
        root: str = DATA_DIR,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            "acm", root=root, fold=fold, batch_size=batch_size, num_workers=num_workers
        )

    def __str__(self) -> str:
        """Return dataset name."""
        return "ACM"


class FreebaseDataModule(HetSheafDataModule):
    """Freebase: Knowledge base subgraph (Book, Author, etc. types)."""

    def __init__(
        self,
        root: str = DATA_DIR,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            "freebase",
            root=root,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def __str__(self) -> str:
        """Return dataset name."""
        return "Freebase"


# ---------------------------------------------------------------------------
# HGBBaseDataModule (legacy / TMP-compatible interface)
# ---------------------------------------------------------------------------


class HGBBaseDataModule(L.LightningDataModule):
    """Legacy HGB DataModule for backward compatibility.

    Used when a more manual configuration of node types and channels is required.
    """

    def __init__(
        self,
        target: str = "author",
        num_classes: int = 4,
        data_dir: str = DATA_DIR,
        task: Literal["multiclass", "multilabel", "binary"] = "multiclass",
        dataset: Literal["IMDB", "DBLP", "ACM", "Freebase"] = "DBLP",
        homogeneous: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.num_classes: int = num_classes
        self.task = task
        self.dataset = dataset
        self.num_nodes = None
        self.in_channels: dict[str, int] | None | int = None
        self.homogeneous = homogeneous
        self.edge_index: dict[str, torch.Tensor] | torch.Tensor | None = None
        self.graph_size: int | None = None
        self.num_node_types: int | None = None
        self.num_edge_types: int | None = None
        self.node_type_names: list[str] | None = None
        self.edge_type_names: list[str] | None = None
        self.metadata = None
        self._pyg_datamodule: Any = None

    def prepare_data(self) -> None:
        """Load the HGB graph and wrap it in a full-batch PyG node DataModule."""
        from torch_geometric.data.lightning import LightningNodeData

        transform = T.Compose(
            [
                T.Constant(node_types=None),
                T.ToUndirected(),
                SubsetTrainSplit(val_ratio=0.1),
                RemoveSelfLoops(),
                T.NormalizeFeatures(),
                T.RemoveDuplicatedEdges(),
            ]
        )
        dataset = HGBDataset(root=self.data_dir, name=self.dataset, transform=transform)
        data: HeteroData | Data = cast(HeteroData, dataset[0]).coalesce()
        input_nodes = data[self.target]

        self.edge_index = data.edge_index_dict
        self.in_channels = {
            node_type: data[node_type].num_features for node_type in data.node_types
        }
        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes

        if self.homogeneous:
            data = data.to_homogeneous()
            self.edge_index = data.edge_index
            input_nodes = data
            assert data.x is not None
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names

        self._pyg_datamodule = LightningNodeData(
            data,
            input_train_nodes=(self.target, input_nodes.train_mask),
            input_val_nodes=(self.target, input_nodes.val_mask),
            input_test_nodes=(self.target, input_nodes.test_mask),
            loader="full",
            batch_size=1,
        )

    def train_dataloader(self) -> Any:
        """Training loader of the wrapped PyG node DataModule."""
        return self._pyg_datamodule.train_dataloader()

    def val_dataloader(self) -> Any:
        """Validation loader of the wrapped PyG node DataModule."""
        return self._pyg_datamodule.val_dataloader()

    def test_dataloader(self) -> Any:
        """Test loader of the wrapped PyG node DataModule."""
        return self._pyg_datamodule.test_dataloader()


# ---------------------------------------------------------------------------
# HGB Link-Prediction datasets (ported verbatim from TMP/datasets/hgb.py)
# ---------------------------------------------------------------------------


class HGBLinkPredBase(L.LightningDataModule):
    """Base class for HGB link-prediction datasets (Amazon, PubMed, YouTube).

    Handles the custom HGB data format (.dat files), downloading from Google Drive,
    and parsing nodes/links into HeteroData.
    """

    gdrive_file_id: str = ""
    zip_name: str = ""
    node_type_map: dict[int, str] = {}
    edge_type_map: dict[int, tuple[str, str, str]] = {}
    target: tuple[str, str, str] = ("", "", "")
    rev_target: tuple[str, str, str] = ("", "", "")
    test_edge_type_id: int = 0

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.homogeneous = homogeneous
        self.split_ratio = split_ratio

        self.metadata = None
        self.in_channels: dict[str, int] | int | None = None
        self.num_nodes: int | None = None
        self.num_node_types: int | None = None
        self.num_edge_types: int | None = None
        self.node_type_names: list[str] | None = None
        self.edge_type_names: list[tuple[str, str, str]] | None = None
        self.graph_size: int | None = None
        self.task = "binary"
        self.num_classes = 1

        self.train_data: Data | HeteroData | None = None
        self.val_data: Data | HeteroData | None = None
        self.test_data: Data | HeteroData | None = None

    def _download_and_extract(self) -> None:
        """Handle GDrive downloading and zip extraction."""
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if not os.path.exists(
            os.path.join(self.data_dir, "node.dat")
        ):  # pragma: no cover
            if not os.path.exists(zip_path):
                print(f"Downloading {self.zip_name} from Google Drive...")
                try:
                    import gdown

                    url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                    gdown.download(url, zip_path, quiet=False)
                except ImportError as err:
                    raise ImportError(
                        "Please install gdown to download HGB datasets: "
                        "pip install gdown"
                    ) from err

            if os.path.exists(zip_path):
                import zipfile

                print(f"Extracting {self.zip_name}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)

                # Flatten directory structure if needed
                for entry in os.listdir(self.data_dir):
                    entry_path = os.path.join(self.data_dir, entry)
                    if os.path.isdir(entry_path) and entry.endswith("_ini"):
                        for fname in os.listdir(entry_path):
                            shutil.move(
                                os.path.join(entry_path, fname),
                                os.path.join(self.data_dir, fname),
                            )
                        os.rmdir(entry_path)
                        break

    def _parse_nodes(
        self,
    ) -> tuple[dict[int, tuple[int, int]], dict[int, list[float]], Counter]:
        """Parse node.dat into type-specific local mappings and features."""
        node_file = os.path.join(self.data_dir, "node.dat")
        global_to_local: dict[int, tuple[int, int]] = {}
        node_features: dict[int, list] = {}
        counts: Counter = Counter()
        type_counters: Counter = Counter()

        with open(node_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                node_id = int(parts[0])
                node_type = int(parts[2])
                global_to_local[node_id] = (node_type, type_counters[node_type])
                type_counters[node_type] += 1
                counts[node_type] += 1
                if len(parts) == 4:
                    node_features[node_id] = [float(x) for x in parts[3].split(",")]

        return global_to_local, node_features, counts

    def _parse_links(
        self,
        filename: str,
        global_to_local: dict[int, tuple[int, int]],
        has_label: bool = False,
    ) -> dict[int, dict]:
        """Parse link.dat into edge lists."""
        link_file = os.path.join(self.data_dir, filename)
        if not os.path.exists(link_file):
            return {}

        edge_data: dict[int, dict] = {}
        with open(link_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])
                weight_or_label = float(parts[3])
                if h_id not in global_to_local or t_id not in global_to_local:
                    continue
                if r_id not in edge_data:
                    edge_data[r_id] = {"src": [], "dst": [], "label": []}
                h_local = global_to_local[h_id][1]
                t_local = global_to_local[t_id][1]
                edge_data[r_id]["src"].append(h_local)
                edge_data[r_id]["dst"].append(t_local)
                if has_label:
                    edge_data[r_id]["label"].append(int(weight_or_label))

        return edge_data

    def _build_hetero_data(
        self,
        global_to_local: dict,
        node_features: dict,
        counts: Counter,
        train_edges: dict[int, dict],
        test_edge_data: dict[int, dict] | None = None,
    ) -> HeteroData:
        """Assemble parsed data into a PyG HeteroData object."""
        data = HeteroData()
        feat_dim = 128
        for type_id, type_name in self.node_type_map.items():
            n = counts[type_id]
            if any(global_to_local[g][0] == type_id for g in node_features):
                local_feats = {}
                for g_id, feat in node_features.items():
                    if global_to_local[g_id][0] == type_id:
                        local_feats[global_to_local[g_id][1]] = feat
                mat = [local_feats[i] for i in range(n)]
                data[type_name].x = torch.tensor(mat, dtype=torch.float)
            else:
                data[type_name].x = torch.ones(n, feat_dim)
            data[type_name].num_nodes = n

        for r_id, edict in train_edges.items():
            if r_id not in self.edge_type_map:
                continue
            src_type, rel_name, dst_type = self.edge_type_map[r_id]
            ei = torch.tensor([edict["src"], edict["dst"]], dtype=torch.long)
            data[src_type, rel_name, dst_type].edge_index = ei
            rev_src, rev_rel, rev_dst = (
                self.rev_target
                if ((src_type, rel_name, dst_type) == self.target)
                else (dst_type, f"rev_{rel_name}", src_type)
            )
            data[rev_src, rev_rel, rev_dst].edge_index = ei.flip(0)

        return data

    def prepare_data(self) -> None:
        """Download, parse, and split the link-prediction dataset."""
        self._download_and_extract()

        global_to_local, node_features, counts = self._parse_nodes()
        train_edges = self._parse_links("link.dat", global_to_local, has_label=False)
        test_edge_data = self._parse_links(
            "link.dat.test", global_to_local, has_label=True
        )

        data = self._build_hetero_data(
            global_to_local, node_features, counts, train_edges, test_edge_data
        )

        self.metadata = data.metadata()
        self.num_nodes = data.num_nodes
        self.num_node_types = len(data.node_types)
        self.num_edge_types = len(data.edge_types)
        self.in_channels = {nt: data[nt].num_features for nt in data.node_types}

        if self.homogeneous:
            data = data.to_homogeneous()
            assert data.x is not None
            self.graph_size = data.x.size(0)
            self.in_channels = data.num_features
            self.num_node_types = data.num_node_types
            self.num_edge_types = data.num_edge_types
            self.node_type_names = data._node_type_names
            self.edge_type_names = data._edge_type_names
        else:
            self.node_type_names = list(data.node_types)
            self.edge_type_names = list(data.edge_types)

        split = T.RandomLinkSplit(
            num_val=self.split_ratio[1],
            num_test=self.split_ratio[2],
            edge_types=None if self.homogeneous else self.target,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.6,
            rev_edge_types=None if self.homogeneous else self.rev_target,
        )
        self.train_data, self.val_data, self.test_data = split(data)

        if self.homogeneous:
            # Ensure graph_size covers all indices in the label index
            max_train = (
                self.train_data.edge_label_index.max().item()
                if self.train_data.edge_label_index.numel() > 0
                else -1
            )
            max_val = (
                self.val_data.edge_label_index.max().item()
                if self.val_data.edge_label_index.numel() > 0
                else -1
            )
            max_test = (
                self.test_data.edge_label_index.max().item()
                if self.test_data.edge_label_index.numel() > 0
                else -1
            )
            assert self.graph_size is not None
            self.graph_size = max(
                self.graph_size, max_train + 1, max_val + 1, max_test + 1
            )

    def train_dataloader(self) -> Any:
        """Full-batch loader over the training link split."""
        assert self.train_data is not None
        return LightningLinkData(self.train_data, loader="full").full_dataloader()

    def val_dataloader(self) -> Any:
        """Full-batch loader over the validation link split."""
        assert self.val_data is not None
        return LightningLinkData(self.val_data, loader="full").full_dataloader()

    def test_dataloader(self) -> Any:
        """Full-batch loader over the test link split."""
        assert self.test_data is not None
        return LightningLinkData(self.test_data, loader="full").full_dataloader()


class HGBAmazonLPDataModule(HGBLinkPredBase):
    """HGB-Amazon link-prediction dataset."""

    gdrive_file_id = "1XD5lXC2NOuu3KLEyYRWad4mqBxZXT468"
    zip_name = "amazon_ini.zip"
    node_type_map = {0: "item"}
    edge_type_map = {
        0: ("item", "also_viewed", "item"),
        1: ("item", "also_bought", "item"),
        2: ("item", "brand", "item"),
        3: ("item", "category", "item"),
    }
    target = ("item", "also_viewed", "item")
    rev_target = ("item", "rev_also_viewed", "item")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(data_dir=f"{data_dir}/hgb_amazon", homogeneous=homogeneous)

    def __repr__(self) -> str:
        """Return dataset name."""
        return "HGBAmazon"


class HGBPubMedLPDataModule(HGBLinkPredBase):
    """HGB-PubMed link-prediction dataset."""

    gdrive_file_id = "1hJHKN5SW8Fv61GRF3Adr6MPOpNA5j0YI"
    zip_name = "PubMed_ini.zip"
    node_type_map = {0: "gene", 1: "disease", 2: "chemical", 3: "species"}
    edge_type_map = {
        0: ("gene", "gene_gene", "gene"),
        1: ("gene", "gene_disease", "disease"),
        2: ("disease", "disease_disease", "disease"),
        3: ("chemical", "chemical_gene", "gene"),
        4: ("chemical", "chemical_disease", "disease"),
        5: ("chemical", "chemical_chemical", "chemical"),
        6: ("chemical", "chemical_species", "species"),
        7: ("species", "species_gene", "gene"),
        8: ("species", "species_disease", "disease"),
        9: ("species", "species_species", "species"),
    }
    target = ("gene", "gene_gene", "gene")
    rev_target = ("gene", "rev_gene_gene", "gene")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(data_dir=f"{data_dir}/hgb_pubmed", homogeneous=homogeneous)

    def __repr__(self) -> str:
        """Return dataset name."""
        return "HGBPubMed"


class HGBYouTubeLPDataModule(HGBLinkPredBase):
    """HGB-YouTube link-prediction dataset."""

    gdrive_file_id = "1JZfztaVNjsBwPkJC2CfaESv-ua78zf-K"
    zip_name = "youtube.zip"
    node_type_map = {0: "user"}
    edge_type_map = {
        0: ("user", "interaction_0", "user"),
        1: ("user", "interaction_1", "user"),
        2: ("user", "interaction_2", "user"),
        3: ("user", "interaction_3", "user"),
        4: ("user", "interaction_4", "user"),
    }
    target = ("user", "interaction_0", "user")
    rev_target = ("user", "rev_interaction_0", "user")
    test_edge_type_id = 0

    def __init__(self, data_dir: str = DATA_DIR, homogeneous: bool = False):
        super().__init__(data_dir=f"{data_dir}/hgb_youtube", homogeneous=homogeneous)

    def _download_and_extract(self) -> None:  # pragma: no cover
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, self.zip_name)

        if not os.path.exists(os.path.join(self.data_dir, "node.dat")):
            if not os.path.exists(zip_path):
                print(f"Downloading {self.zip_name} from Google Drive...")
                try:
                    import gdown

                    url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                    gdown.download(url, zip_path, quiet=False)
                except ImportError as err:
                    raise ImportError(
                        "Please install gdown to download HGB datasets: "
                        "pip install gdown"
                    ) from err

            if os.path.exists(zip_path):
                import zipfile

                print(f"Extracting {self.zip_name}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)

                for entry in os.listdir(self.data_dir):
                    entry_path = os.path.join(self.data_dir, entry)
                    if os.path.isdir(entry_path) and entry not in (".", ".."):
                        candidate = os.path.join(entry_path, "node.dat")
                        if os.path.exists(candidate):
                            for fname in os.listdir(entry_path):
                                shutil.move(
                                    os.path.join(entry_path, fname),
                                    os.path.join(self.data_dir, fname),
                                )
                            os.rmdir(entry_path)
                            break

    def __repr__(self) -> str:
        """Return dataset name."""
        return "HGBYouTube"
