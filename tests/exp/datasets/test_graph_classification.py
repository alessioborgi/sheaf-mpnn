# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/datasets/graph_classification.py - TUDataset DataModules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from exp.data import DatasetInfo
from exp.datasets.graph_classification import (
    COLLABDataModule,
    DDDataModule,
    ENZYMESDataModule,
    ExtractTypes,
    GCHetSheafDataModule,
    GCSheafDataModule,
    GraphClassificationDataModule,
    IMDBBinaryDataModule,
    IMDBMultiDataModule,
    MUTAGDataModule,
    NCI1DataModule,
    NCI109DataModule,
    PROTEINSDataModule,
    PTCMRDataModule,
    REDDIT5KDataModule,
    REDDITBinaryDataModule,
    SetHomogeneous,
    gc_het_registry,
    gc_homo_registry,
)
from exp.datasets.info import GCDatasetInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    num_nodes: int = 6,
    num_node_types: int = 2,
    num_edge_types: int = 2,
    num_classes: int = 2,
) -> Data:
    """Single synthetic heterogeneous graph with typed node_type, edge_type."""
    num_edges = num_nodes * 2
    return Data(
        x=torch.ones(num_nodes, 4, dtype=torch.float),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.randint(0, num_classes, (1,)),
        node_type=torch.randint(0, num_node_types, (num_nodes,)),
        edge_type=torch.randint(0, num_edge_types, (num_edges,)),
    )


def _make_homo_graph(num_nodes: int = 6, num_classes: int = 2) -> Data:
    """Single synthetic homogeneous graph
    (all types zero, as SetHomogeneous produces).
    """
    num_edges = num_nodes * 2
    return Data(
        x=torch.ones(num_nodes, 1, dtype=torch.float),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.randint(0, num_classes, (1,)),
        node_type=torch.zeros(num_nodes, dtype=torch.long),
        edge_type=torch.zeros(num_edges, dtype=torch.long),
    )


def _make_mock_tu_dataset(
    n: int = 20,
    num_classes: int = 2,
    num_node_features: int = 4,
    homogeneous: bool = False,
):
    """Mock TUDataset-like object with n graphs."""
    graphs = [_make_homo_graph() if homogeneous else _make_graph() for _ in range(n)]

    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=n)
    mock_ds.__getitem__ = MagicMock(side_effect=lambda idx: graphs[idx])
    mock_ds.num_classes = num_classes
    mock_ds.num_node_features = num_node_features

    # Support slicing (called when dataset[perm[:n_train]])
    def _getitem(idx):
        if isinstance(idx, torch.Tensor):
            sub = MagicMock()
            items = [graphs[i] for i in idx.tolist()]
            sub.__len__ = MagicMock(return_value=len(items))
            sub.__getitem__ = MagicMock(side_effect=lambda i: items[i])
            sub.__iter__ = MagicMock(return_value=iter(items))
            return sub
        return graphs[idx]

    mock_ds.__getitem__ = MagicMock(side_effect=_getitem)
    mock_ds.__iter__ = MagicMock(return_value=iter(graphs))
    return mock_ds


# ---------------------------------------------------------------------------
# TestGCDatasetInfo
# ---------------------------------------------------------------------------


class TestGCDatasetInfo:
    def test_construction_and_fields(self):
        info = GCDatasetInfo(
            name="mutag",
            num_features=4,
            num_classes=2,
            num_splits=10,
            metric="acc",
            split_type="seed",
            num_node_types=7,
            num_edge_types=4,
            task="binary",
        )
        assert info.name == "mutag", f"Expected name mutag, got {info.name}"
        assert info.num_node_types == 7, (
            f"Expected 7 node types, got {info.num_node_types}"
        )
        assert info.num_edge_types == 4, (
            f"Expected 4 edge types, got {info.num_edge_types}"
        )
        assert info.task == "binary", f"Expected task binary, got {info.task}"

    def test_is_dataset_info_subclass(self):
        info = GCDatasetInfo(
            name="mutag",
            num_features=4,
            num_classes=2,
            num_splits=10,
            metric="acc",
            split_type="seed",
        )
        assert isinstance(info, DatasetInfo), "info should be a DatasetInfo"

    def test_frozen(self):
        info = GCDatasetInfo(
            name="mutag",
            num_features=4,
            num_classes=2,
            num_splits=10,
            metric="acc",
            split_type="seed",
        )
        with pytest.raises((AttributeError, TypeError)):
            info.num_classes = 99  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TestRegistries
# ---------------------------------------------------------------------------


class TestRegistries:
    @pytest.mark.parametrize(
        "name", ["collab", "imdb_b", "imdb_m", "reddit_b", "reddit_5k"]
    )
    def test_homo_registry_contains_all_names(self, name: str):
        assert name in gc_homo_registry, f"Dataset {name} missing from homo registry"

    @pytest.mark.parametrize(
        "name", ["mutag", "proteins", "enzymes", "nci1", "nci109", "ptcmr", "dd"]
    )
    def test_het_registry_contains_all_names(self, name: str):
        assert name in gc_het_registry, f"Dataset {name} missing from het registry"

    def test_mutag_has_correct_type_cols(self):
        entry = gc_het_registry["mutag"]
        assert entry.node_type_cols == [
            0,
            7,
        ], f"Expected node_type_cols [0, 7], got {entry.node_type_cols}"
        assert entry.edge_type_cols == [
            0,
            4,
        ], f"Expected edge_type_cols [0, 4], got {entry.edge_type_cols}"

    def test_proteins_has_no_edge_type_cols(self):
        entry = gc_het_registry["proteins"]
        assert entry.node_type_cols is not None, (
            "Proteins should have node_type_cols defined"
        )
        assert entry.edge_type_cols is None, "Proteins should not have edge_type_cols"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            _ = gc_het_registry["nonexistent"]


# ---------------------------------------------------------------------------
# TestExtractTypes
# ---------------------------------------------------------------------------


class TestExtractTypes:
    def test_extracts_node_type_from_one_hot(self):
        # 7-col one-hot: first atom type for all nodes
        x = torch.eye(7).repeat(3, 1)  # 21 nodes, 7 cols
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = 21

        transform = ExtractTypes(node_type_cols=[0, 7], edge_type_cols=None)
        out = transform(data)

        assert hasattr(out, "node_type"), "Output should have node_type attribute"
        assert out.node_type.dtype == torch.long, (
            f"Expected node_type dtype long, got {out.node_type.dtype}"
        )
        assert out.node_type.shape == (21,), (
            f"Expected shape (21,), got {out.node_type.shape}"
        )

    def test_extracts_edge_type_from_one_hot(self):
        x = torch.ones(4, 1)
        # 4 edges with one-hot edge attrs in 4 cols
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.eye(4)  # 4 edges, 4 bond types
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 4

        transform = ExtractTypes(node_type_cols=None, edge_type_cols=[0, 4])
        out = transform(data)

        assert hasattr(out, "edge_type"), "Output should have edge_type attribute"
        assert out.edge_type.dtype == torch.long, (
            f"Expected edge_type dtype long, got {out.edge_type.dtype}"
        )
        assert out.edge_type.shape == (4,), (
            f"Expected shape (4,), got {out.edge_type.shape}"
        )

    def test_creates_constant_features_when_x_none(self):
        data = Data(
            x=None,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        data.num_nodes = 5

        transform = ExtractTypes(node_type_cols=None, edge_type_cols=None)
        out = transform(data)

        assert out.x is not None, "Output x should not be None"
        assert out.x.shape == (5, 1), f"Expected shape (5, 1), got {out.x.shape}"
        assert torch.all(out.x == 1.0), "Output x should be all ones"

    def test_creates_degree_features_when_x_none(self):
        data = Data(
            x=None,
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        data.num_nodes = 3

        transform = ExtractTypes(node_feature_type="degree")
        out = transform(data)

        assert out.x is not None, "Output x should not be None (degree)"
        assert out.x.shape == (3, 1), f"Expected shape (3, 1), got {out.x.shape}"

    def test_homogeneous_types_when_no_cols(self):
        data = Data(
            x=torch.ones(5, 3),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        data.num_nodes = 5

        transform = ExtractTypes(node_type_cols=None, edge_type_cols=None)
        out = transform(data)

        assert torch.all(out.node_type == 0), "node_type should be all zeros"


# ---------------------------------------------------------------------------
# TestSetHomogeneous
# ---------------------------------------------------------------------------


class TestSetHomogeneous:
    def test_sets_all_types_to_zero(self):
        data = Data(
            x=torch.ones(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        )
        data.num_nodes = 4

        out = SetHomogeneous()(data)

        assert torch.all(out.node_type == 0), "node_type should be all zeros"
        assert torch.all(out.edge_type == 0), "edge_type should be all zeros"

    def test_creates_x_when_none(self):
        data = Data(
            x=None,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        data.num_nodes = 3

        out = SetHomogeneous()(data)
        assert out.x is not None, "Output x should not be None after SetHomogeneous"


# ---------------------------------------------------------------------------
# TestGCSheafDataModule
# ---------------------------------------------------------------------------


class TestGCSheafDataModule:
    @pytest.fixture
    def mock_tu(self):
        # homogeneous=True: graphs have all-zero node_type/edge_type
        # (as SetHomogeneous produces)
        return _make_mock_tu_dataset(n=20, homogeneous=True)

    def test_setup_creates_splits(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        assert dm._train_dataset is not None, (
            "Train dataset was not initialized in setup"
        )
        assert dm._val_dataset is not None, "Val dataset was not initialized in setup"
        assert dm._test_dataset is not None, "Test dataset was not initialized in setup"

    def test_split_sizes_sum_to_dataset_size(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        n_train, n_val, n_test = dm.split_sizes
        assert n_train + n_val + n_test == 20, (
            f"Split sizes {dm.split_sizes} do not sum to 20"
        )

    def test_split_sizes_are_positive_ints(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        for count in dm.split_sizes:
            assert isinstance(count, int), f"Split size {count} should be an integer"
            assert count > 0, f"Split size {count} should be positive"

    def test_setup_calls_load_once(self, mock_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_tu
        ) as mock_cls:
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        mock_cls.assert_called_once()

    def test_info_fields(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        info = dm.info
        assert isinstance(info, GCDatasetInfo), "dm.info should be GCDatasetInfo"
        assert info.split_type == "seed", (
            f"Expected split_type seed, got {info.split_type}"
        )
        assert info.num_splits == 10, f"Expected 10 splits, got {info.num_splits}"
        assert info.num_node_types == 1, (
            f"Expected 1 node type, got {info.num_node_types}"
        )
        assert info.num_edge_types == 1, (
            f"Expected 1 edge type, got {info.num_edge_types}"
        )

    def test_info_is_dataset_info_subclass(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        assert isinstance(dm.info, DatasetInfo), "dm.info should be a DatasetInfo"
        assert isinstance(dm.info, GCDatasetInfo), "dm.info should be a GCDatasetInfo"

    def test_setup_valid_with_different_folds(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm0 = GCSheafDataModule("collab", root="/tmp/fake", fold=0)
            dm0.setup()
            dm1 = GCSheafDataModule("collab", root="/tmp/fake", fold=1)
            dm1.setup()

        # Different folds should (almost certainly) produce different train sizes
        # or at least the permutation is different.  We just verify both produce
        # valid sizes summing to 20.
        assert sum(dm0.split_sizes) == 20, (
            f"Fold 0 split sizes {dm0.split_sizes} do not sum to 20"
        )
        assert sum(dm1.split_sizes) == 20, (
            f"Fold 1 split sizes {dm1.split_sizes} do not sum to 20"
        )

    def test_num_nodes_raises(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()

        with pytest.raises(AttributeError):
            _ = dm.num_nodes

    def test_batch_size_propagation(self, mock_tu):
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake", batch_size=8)
            dm.setup()
            dl = dm.train_dataloader()
            assert dl.batch_size == 8


# ---------------------------------------------------------------------------
# TestGCHetSheafDataModule
# ---------------------------------------------------------------------------


class TestGCHetSheafDataModule:
    @pytest.fixture
    def mock_het_tu(self):
        # Graphs with 2 node types and 2 edge types
        return _make_mock_tu_dataset(n=20, num_classes=2, num_node_features=1)

    def test_setup_creates_splits(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        assert dm._train_dataset is not None, (
            "Train dataset was not initialized in setup (het)"
        )
        assert dm._val_dataset is not None, (
            "Val dataset was not initialized in setup (het)"
        )
        assert dm._test_dataset is not None, (
            "Test dataset was not initialized in setup (het)"
        )

    def test_split_sizes_sum_to_dataset_size(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        assert sum(dm.split_sizes) == 20, (
            f"Split sizes {dm.split_sizes} do not sum to 20 (het)"
        )

    def test_setup_uses_extract_types_transform(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ) as mock_cls:
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        _, kwargs = mock_cls.call_args
        transform = kwargs.get("transform") or mock_cls.call_args[0][2]
        assert isinstance(transform, ExtractTypes), (
            "TUDataset should be initialized with ExtractTypes transform"
        )

    def test_num_node_types_property(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        assert dm.num_node_types >= 1, (
            f"Expected at least one node type, got {dm.num_node_types}"
        )

    def test_num_edge_types_property(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        assert dm.num_edge_types >= 1, (
            f"Expected at least one edge type, got {dm.num_edge_types}"
        )

    def test_num_node_types_before_setup_raises(self):
        dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_node_types

    def test_info_is_dataset_info_subclass(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()

        assert isinstance(dm.info, DatasetInfo), (
            "dm.info should be an instance of DatasetInfo"
        )
        assert isinstance(dm.info, GCDatasetInfo), (
            "dm.info should be an instance of GCDatasetInfo"
        )

    def test_setup_valid_with_different_folds(self, mock_het_tu):
        with patch(
            "exp.datasets.graph_classification.TUDataset", return_value=mock_het_tu
        ):
            dm0 = GCHetSheafDataModule("mutag", root="/tmp/fake", fold=0)
            dm0.setup()
            dm1 = GCHetSheafDataModule("mutag", root="/tmp/fake", fold=1)
            dm1.setup()

        assert sum(dm0.split_sizes) == 20, (
            f"Fold 0 split sizes {dm0.split_sizes} do not sum to 20"
        )
        assert sum(dm1.split_sizes) == 20, (
            f"Fold 1 split sizes {dm1.split_sizes} do not sum to 20"
        )


# ---------------------------------------------------------------------------
# TestExtractTypes -- missing branches
# ---------------------------------------------------------------------------


class TestExtractTypesMissingBranches:
    def test_degree_fallback_when_features_are_entirely_one_hot(self):
        """Lines 99-105: node_type_cols covers all columns -> degree fallback."""
        # 7-col one-hot features, all cols are types (start=0, end=7=size)
        x = torch.eye(7).repeat(3, 1)  # 21 nodes, 7 cols
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = 21

        transform = ExtractTypes(node_type_cols=[0, 7], node_feature_type="degree")
        out = transform(data)
        # x should now be degree features (shape [21, 1])
        assert out.x is not None
        assert out.x.shape == (
            21,
            1,
        ), f"Expected degree fallback shape (21,1), got {out.x.shape}"

    def test_edge_type_1d_categorical(self):
        """Line 129: edge_attr is 1D categorical -> edge_type = edge_attr.long()."""
        x = torch.ones(4, 1)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.tensor([0, 1, 2, 3], dtype=torch.float)  # 1D
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 4

        transform = ExtractTypes(edge_type_cols=[0, 4])
        out = transform(data)
        assert out.edge_type is not None
        assert out.edge_type.dtype == torch.long
        assert torch.equal(out.edge_type, torch.tensor([0, 1, 2, 3]))

    def test_edge_type_2d_all_cols_replaced_with_constant(self):
        """Lines 133-137: start==0 and end>=size -> replace edge_attr with ones."""
        x = torch.ones(4, 1)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.eye(4)  # 4 edges, 4 cols; all cols are type
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 4

        transform = ExtractTypes(edge_type_cols=[0, 4])
        out = transform(data)
        # edge_attr replaced with constant 1s
        assert out.edge_attr is not None
        assert out.edge_attr.shape == (4, 1)
        assert torch.all(out.edge_attr == 1.0)

    def test_edge_type_2d_mid_cols_sliced(self):
        """Lines 138-145: various edge_attr slicing branches."""
        # start=0, end < size: keep columns from end onwards
        x = torch.ones(2, 1)
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[1.0, 0.0, 5.0]])  # 1 edge, 3 cols; types at [0:2]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 2

        transform = ExtractTypes(edge_type_cols=[0, 2])
        out = transform(data)
        # Remaining col should be [5.0]
        assert out.edge_attr is not None
        assert out.edge_attr.shape == (1, 1)
        assert out.edge_attr[0, 0] == pytest.approx(5.0)

    def test_edge_type_2d_tail_cols_sliced(self):
        """End >= size and start > 0: keep columns up to start."""
        x = torch.ones(2, 1)
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[5.0, 1.0, 0.0]])  # type cols at [1:3]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 2

        transform = ExtractTypes(edge_type_cols=[1, 3])
        out = transform(data)
        # Remaining col should be [5.0]
        assert out.edge_attr is not None
        assert out.edge_attr.shape == (1, 1)
        assert out.edge_attr[0, 0] == pytest.approx(5.0)

    def test_edge_type_2d_middle_cols_sliced(self):
        """Lines 143-144: start > 0 and end < size -> cat before and after."""
        x = torch.ones(2, 1)
        edge_index = torch.tensor([[0], [1]])
        edge_attr = torch.tensor([[3.0, 1.0, 0.0, 7.0]])  # type cols at [1:3]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 2

        transform = ExtractTypes(edge_type_cols=[1, 3])
        out = transform(data)
        # Remaining cols should be [3.0, 7.0]
        assert out.edge_attr is not None
        assert out.edge_attr.shape == (1, 2)
        assert out.edge_attr[0, 0] == pytest.approx(3.0)
        assert out.edge_attr[0, 1] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# TestSetHomogeneous -- missing branches
# ---------------------------------------------------------------------------


class TestSetHomogeneousMissingBranches:
    def test_degree_features_when_x_is_none(self):
        """Lines 181-185: SetHomogeneous with degree features when x=None."""
        data = Data(
            x=None,
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        data.num_nodes = 3
        out = SetHomogeneous(node_feature_type="degree")(data)
        assert out.x is not None
        assert out.x.shape == (3, 1), f"Expected shape (3, 1), got {out.x.shape}"

    def test_edge_type_none_when_edge_index_is_none(self):
        """Line 194: edge_type=None when edge_index is None (PyG ignores None)."""
        data = Data(x=torch.ones(3, 2), edge_index=None)
        data.num_nodes = 3
        out = SetHomogeneous()(data)
        # PyG Data ignores None assignments, so edge_type is absent (not None)
        assert not hasattr(out, "edge_type") or out.get("edge_type") is None


# ---------------------------------------------------------------------------
# TestGCSheafDataModule -- missing branches
# ---------------------------------------------------------------------------


class TestGCSheafDataModuleMissingBranches:
    def test_split_sizes_before_setup_raises(self):
        """Line 377: RuntimeError when split_sizes accessed before setup()."""
        dm = GCSheafDataModule("collab", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.split_sizes

    def test_num_edges_raises(self):
        """Line 394-396: num_edges raises AttributeError."""
        dm = GCSheafDataModule("collab", root="/tmp/fake")
        with pytest.raises(AttributeError):
            _ = dm.num_edges

    def test_homophily_raises(self):
        """Homophily raises AttributeError."""
        dm = GCSheafDataModule("collab", root="/tmp/fake")
        with pytest.raises(AttributeError):
            _ = dm.homophily

    def test_val_and_test_dataloaders(self):
        """Lines 587-588, 598-599: val and test dataloaders work."""
        mock_tu = _make_mock_tu_dataset(n=20, homogeneous=True)
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        assert val_dl is not None
        assert test_dl is not None


# ---------------------------------------------------------------------------
# TestGCHetSheafDataModule -- missing branches
# ---------------------------------------------------------------------------


class TestGCHetSheafDataModuleMissingBranches:
    def test_num_edge_types_before_setup_raises(self):
        """Line 482: RuntimeError when num_edge_types accessed before setup()."""
        dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_edge_types


# ---------------------------------------------------------------------------
# TestGraphClassificationDataModule (legacy)
# ---------------------------------------------------------------------------


class TestGraphClassificationDataModule:
    @pytest.fixture
    def mock_tu(self):
        return _make_mock_tu_dataset(n=20, homogeneous=True)

    def test_init_covers_constructor(self, mock_tu):
        """Lines 515-524: constructor sets all fields correctly."""
        dm = GraphClassificationDataModule(
            root="/tmp/fake",
            name="MUTAG",
            homogeneous=True,
            node_type_cols=None,
            edge_type_cols=None,
            node_feature_type="constant",
            num_workers=0,
            train_split=0.8,
            val_split=0.1,
            seed=42,
            batch_size=2,
        )
        assert dm._homogeneous is True
        assert dm._train_split == 0.8
        assert dm._val_split == 0.1

    def test_build_transform_homogeneous(self):
        """Lines 527-528: homogeneous=True -> SetHomogeneous."""
        dm = GraphClassificationDataModule(root="/tmp", name="X", homogeneous=True)
        t = dm._build_transform()
        assert isinstance(t, SetHomogeneous)

    def test_build_transform_heterogeneous(self):
        """Lines 529-533: homogeneous=False -> ExtractTypes."""
        dm = GraphClassificationDataModule(
            root="/tmp",
            name="X",
            homogeneous=False,
            node_type_cols=[0, 3],
            edge_type_cols=[0, 2],
        )
        t = dm._build_transform()
        assert isinstance(t, ExtractTypes)

    def test_get_tu_name_returns_name(self):
        """Line 536: _get_tu_name() returns self._name."""
        dm = GraphClassificationDataModule(root="/tmp", name="MUTAG")
        assert dm._get_tu_name() == "MUTAG"

    def test_get_metric_returns_acc(self):
        """Line 539: _get_metric() always returns 'acc'."""
        dm = GraphClassificationDataModule(root="/tmp", name="MUTAG")
        assert dm._get_metric() == "acc"

    def test_setup_creates_splits(self, mock_tu):
        """Lines 542-573: setup() creates train/val/test datasets."""
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GraphClassificationDataModule(root="/tmp/fake", name="MUTAG")
            dm.setup()
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert dm._test_dataset is not None

    def test_dataloaders(self, mock_tu):
        """Lines 575-606: train/val/test dataloaders work."""
        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_tu):
            dm = GraphClassificationDataModule(root="/tmp/fake", name="MUTAG")
            dm.setup()
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None


# ---------------------------------------------------------------------------
# TestConcreteShims -- cover the super().__init__() call in each concrete class
# ---------------------------------------------------------------------------


class TestConcreteShimsInit:
    """Instantiating each concrete class covers the super().__init__() call."""

    def test_mutag_init(self):
        dm = MUTAGDataModule(root="/tmp/fake")
        assert dm._name == "mutag"

    def test_proteins_init(self):
        dm = PROTEINSDataModule(root="/tmp/fake")
        assert dm._name == "proteins"

    def test_enzymes_init(self):
        dm = ENZYMESDataModule(root="/tmp/fake")
        assert dm._name == "enzymes"

    def test_nci1_init(self):
        dm = NCI1DataModule(root="/tmp/fake")
        assert dm._name == "nci1"

    def test_nci109_init(self):
        dm = NCI109DataModule(root="/tmp/fake")
        assert dm._name == "nci109"

    def test_ptcmr_init(self):
        dm = PTCMRDataModule(root="/tmp/fake")
        assert dm._name == "ptcmr"

    def test_dd_init(self):
        dm = DDDataModule(root="/tmp/fake")
        assert dm._name == "dd"

    def test_collab_init(self):
        dm = COLLABDataModule(root="/tmp/fake")
        assert dm._name == "collab"

    def test_imdb_binary_init(self):
        dm = IMDBBinaryDataModule(root="/tmp/fake")
        assert dm._name == "imdb_b"

    def test_imdb_multi_init(self):
        dm = IMDBMultiDataModule(root="/tmp/fake")
        assert dm._name == "imdb_m"

    def test_reddit_binary_init(self):
        dm = REDDITBinaryDataModule(root="/tmp/fake")
        assert dm._name == "reddit_b"

    def test_reddit_5k_init(self):
        dm = REDDIT5KDataModule(root="/tmp/fake")
        assert dm._name == "reddit_5k"
