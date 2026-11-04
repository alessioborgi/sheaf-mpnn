# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/datasets/hgb.py - HGB DataModules and registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from exp.data import DatasetInfo
from exp.datasets.hgb import (
    ACMDataModule,
    DBLPDataModule,
    FreebaseDataModule,
    HetSheafDataModule,
    HGBAmazonLPDataModule,
    HGBBaseDataModule,
    HGBLinkPredBase,
    HGBPubMedLPDataModule,
    HGBYouTubeLPDataModule,
    IMDBDataModule,
    _load_hgb_dataset,
    het_dataset_registry,
)
from exp.datasets.info import HetDatasetInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_homo_data(
    num_nodes: int = 30,
    num_features: int = 8,
    num_node_types: int = 2,
    num_edge_types: int = 3,
) -> Data:
    """Synthetic homogeneous Data as returned by _load_hgb_dataset."""
    num_edges = num_nodes * 2
    # Build masks: first half = target node type; give each split some nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    val_mask[10:15] = True
    test_mask[15:20] = True

    return Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.randint(0, 4, (num_nodes,)),
        node_type=torch.randint(0, num_node_types, (num_nodes,)),
        edge_type=torch.randint(0, num_edge_types, (num_edges,)),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _make_het_info(
    name: str = "dblp",
    num_features: int = 8,
    num_node_types: int = 2,
    num_edge_types: int = 3,
) -> HetDatasetInfo:
    return HetDatasetInfo(
        name=name,
        num_features=num_features,
        num_classes=4,
        num_splits=1,
        metric="acc",
        split_type="hgb_builtin",
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        node_type_names=("author", "paper"),
        edge_type_names=("author__cites__paper",),
        task="multiclass",
    )


# ---------------------------------------------------------------------------
# TestHetDatasetInfo
# ---------------------------------------------------------------------------


class TestHetDatasetInfo:
    def test_construction_and_fields(self):
        info = _make_het_info()
        assert info.name == "dblp", f"Expected name dblp, got {info.name}"
        assert info.num_features == 8, f"Expected 8 features, got {info.num_features}"
        assert info.num_classes == 4, f"Expected 4 classes, got {info.num_classes}"
        assert info.num_node_types == 2, (
            f"Expected 2 node types, got {info.num_node_types}"
        )
        assert info.num_edge_types == 3, (
            f"Expected 3 edge types, got {info.num_edge_types}"
        )
        assert info.task == "multiclass", f"Expected task multiclass, got {info.task}"
        assert info.split_type == "hgb_builtin", (
            f"Expected split_type hgb_builtin, got {info.split_type}"
        )

    def test_is_dataset_info_subclass(self):
        info = _make_het_info()
        assert isinstance(info, DatasetInfo), "info should be a DatasetInfo"

    def test_frozen(self):
        info = _make_het_info()
        with pytest.raises((AttributeError, TypeError)):
            info.num_classes = 99  # ty: ignore[invalid-assignment]

    def test_base_fields_accessible(self):
        info = _make_het_info()
        # These are the fields module.py reads on DatasetInfo
        assert hasattr(info, "metric"), "info should have metric"
        assert hasattr(info, "num_features"), "info should have num_features"
        assert hasattr(info, "num_classes"), "info should have num_classes"


# ---------------------------------------------------------------------------
# TestHetDatasetRegistry
# ---------------------------------------------------------------------------


class TestHetDatasetRegistry:
    @pytest.mark.parametrize("name", ["imdb", "dblp", "acm", "freebase"])
    def test_all_names_present(self, name: str):
        assert name in het_dataset_registry, f"Dataset {name} missing from registry"

    @pytest.mark.parametrize(
        "name,expected_classes,expected_target,expected_task",
        [
            ("imdb", 5, "movie", "multilabel"),
            ("dblp", 4, "author", "multiclass"),
            ("acm", 3, "paper", "multiclass"),
            ("freebase", 7, "book", "multiclass"),
        ],
    )
    def test_entry_metadata(
        self, name: str, expected_classes: int, expected_target: str, expected_task: str
    ):
        entry = het_dataset_registry[name]
        assert entry.num_classes == expected_classes, (
            f"Expected {expected_classes} classes for {name}, got {entry.num_classes}"
        )
        assert entry.target == expected_target, (
            f"Expected target {expected_target} for {name}, got {entry.target}"
        )
        assert entry.task == expected_task, (
            f"Expected task {expected_task} for {name}, got {entry.task}"
        )

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            _ = het_dataset_registry["nonexistent_dataset"]

    def test_entry_is_frozen(self):
        entry = het_dataset_registry["dblp"]
        with pytest.raises((AttributeError, TypeError)):
            entry.num_classes = 99  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TestLoadHgbDataset
# ---------------------------------------------------------------------------


class TestLoadHgbDataset:
    def _mock_hgb_dataset(self):
        """Return a mock object that simulates HGBDataset[0].coalesce().

        SubsetTrainSplit (applied as a transform) creates val_mask from train_mask,
        so the mock must already carry all three masks as the transform would set them.
        """
        from torch_geometric.data import HeteroData

        het = HeteroData()
        het["author"].x = torch.randn(10, 4)
        het["author"].num_nodes = 10
        het["paper"].x = torch.randn(15, 4)
        het["paper"].num_nodes = 15
        # train_mask / val_mask / test_mask as SubsetTrainSplit would produce
        het["author"].train_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].train_mask[:5] = True
        het["author"].val_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].val_mask[5:7] = True
        het["author"].test_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].test_mask[8:] = True
        het["author", "cites", "paper"].edge_index = torch.randint(0, 10, (2, 20))
        return het

    def test_returns_data_and_info(self):
        het = self._mock_hgb_dataset()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            data, info = _load_hgb_dataset("dblp", root="/tmp/fake")

        assert isinstance(data, Data), "Expected output data to be a Data object"
        assert isinstance(info, HetDatasetInfo), (
            "Expected output info to be HetDatasetInfo"
        )

    def test_data_has_node_type_and_edge_type(self):
        het = self._mock_hgb_dataset()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            data, _ = _load_hgb_dataset("dblp", root="/tmp/fake")

        assert hasattr(data, "node_type"), "Data should have node_type"
        assert data.node_type is not None, "node_type should not be None"
        assert data.node_type.dtype == torch.long, (
            f"Expected node_type dtype long, got {data.node_type.dtype}"
        )

    def test_data_has_masks(self):
        het = self._mock_hgb_dataset()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            data, _ = _load_hgb_dataset("dblp", root="/tmp/fake")

        assert hasattr(data, "train_mask"), "Data should have train_mask"
        assert hasattr(data, "test_mask"), "Data should have test_mask"

    def test_info_has_hgb_split_type(self):
        het = self._mock_hgb_dataset()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            _, info = _load_hgb_dataset("dblp", root="/tmp/fake")

        assert info.split_type == "hgb_builtin", (
            f"Expected hgb_builtin split_type, got {info.split_type}"
        )
        assert info.num_splits == 1, f"Expected 1 split, got {info.num_splits}"

    def test_info_has_type_counts(self):
        het = self._mock_hgb_dataset()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            _, info = _load_hgb_dataset("dblp", root="/tmp/fake")

        assert info.num_node_types >= 1, (
            f"Expected at least one node type, got {info.num_node_types}"
        )
        assert info.num_edge_types >= 1, (
            f"Expected at least one edge type, got {info.num_edge_types}"
        )


# ---------------------------------------------------------------------------
# TestHetSheafDataModule
# ---------------------------------------------------------------------------


class TestHetSheafDataModule:
    @pytest.fixture
    def mock_load(self):
        data = _make_homo_data()
        info = _make_het_info()
        return data, info

    def test_setup_sets_data_split_info(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert dm._data is not None, "Data was not loaded in setup"
        assert dm._split is not None, "Split was not initialized in setup"
        assert dm._info is not None, "Info was not initialized in setup"
        # For HGB the single fixed split means _split is _data
        assert dm._split is dm._data, "Split should be the same as data for HGB"

    def test_setup_calls_load_once(self, mock_load):
        data, info = mock_load
        with patch(
            "exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)
        ) as mock_fn:
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        mock_fn.assert_called_once()

    def test_info_before_setup_raises(self):
        dm = HetSheafDataModule("dblp", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.info

    def test_num_node_types_before_setup_raises(self):
        dm = HetSheafDataModule("dblp", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_node_types

    def test_num_nodes(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert dm.num_nodes == data.num_nodes, (
            f"Expected {data.num_nodes} nodes, got {dm.num_nodes}"
        )

    def test_num_edges(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert dm.num_edges == data.num_edges // 2, (
            f"Expected {data.num_edges // 2} edges, got {dm.num_edges}"
        )

    def test_num_node_types(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert dm.num_node_types == info.num_node_types, (
            f"Expected {info.num_node_types} node types, got {dm.num_node_types}"
        )

    def test_num_edge_types(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert dm.num_edge_types == info.num_edge_types, (
            f"Expected {info.num_edge_types} edge types, got {dm.num_edge_types}"
        )

    def test_split_sizes_returns_mask_counts(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        n_train, n_val, n_test = dm.split_sizes
        assert n_train == int(data.train_mask.sum()), (
            f"Expected {int(data.train_mask.sum())} train nodes, got {n_train}"
        )
        assert n_val == int(data.val_mask.sum()), (
            f"Expected {int(data.val_mask.sum())} val nodes, got {n_val}"
        )
        assert n_test == int(data.test_mask.sum()), (
            f"Expected {int(data.test_mask.sum())} test nodes, got {n_test}"
        )

    def test_batch_has_node_type_and_edge_type(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert hasattr(batch, "node_type"), "Batch missing node_type"
        assert hasattr(batch, "edge_type"), "Batch missing edge_type"

    def test_all_dataloaders_same_node_count(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()))
        test_batch = next(iter(dm.test_dataloader()))
        # All three loaders return the full graph
        assert train_batch.num_nodes == val_batch.num_nodes == test_batch.num_nodes, (
            "Dataloaders returned batches with different node counts: "
            f"{train_batch.num_nodes}, {val_batch.num_nodes}, {test_batch.num_nodes}"
        )

    def test_info_is_dataset_info_subclass(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            dm.setup()

        assert isinstance(dm.info, DatasetInfo), (
            "dm.info should be an instance of DatasetInfo"
        )
        assert isinstance(dm.info, HetDatasetInfo), (
            "dm.info should be an instance of HetDatasetInfo"
        )

    def test_fold_parameter_accepted(self, mock_load):
        data, info = mock_load
        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake", fold=5)
            dm.setup()

        # fold is silently ignored for HGB (single fixed split)
        assert dm._data is not None, "Data was not loaded when fold is provided"


# ---------------------------------------------------------------------------
# TestHetSheafDataModule -- missing branches
# ---------------------------------------------------------------------------


class TestHetSheafDataModuleMissingBranches:
    def test_num_edge_types_before_setup_raises(self):
        """num_edge_types before setup() raises RuntimeError."""
        dm = HetSheafDataModule("dblp", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_edge_types


# ---------------------------------------------------------------------------
# TestConcreteHetShims -- cover super().__init__() in each concrete class
# ---------------------------------------------------------------------------


class TestConcreteHetShimsInit:
    def test_imdb_init(self):
        dm = IMDBDataModule(root="/tmp/fake")
        assert dm._name == "imdb"
        assert str(dm) == "IMDB"

    def test_dblp_init(self):
        dm = DBLPDataModule(root="/tmp/fake")
        assert dm._name == "dblp"
        assert str(dm) == "DBLP"

    def test_acm_init(self):
        dm = ACMDataModule(root="/tmp/fake")
        assert dm._name == "acm"
        assert str(dm) == "ACM"

    def test_freebase_init(self):
        dm = FreebaseDataModule(root="/tmp/fake")
        assert dm._name == "freebase"
        assert str(dm) == "Freebase"


# ---------------------------------------------------------------------------
# TestHGBBaseDataModule -- constructor coverage
# ---------------------------------------------------------------------------


class TestHGBBaseDataModule:
    def test_init_sets_fields(self):
        """Constructor stores all passed fields correctly."""
        dm = HGBBaseDataModule(
            target="author",
            num_classes=4,
            data_dir="/tmp/fake",
            task="multiclass",
            dataset="DBLP",
            homogeneous=False,
        )
        assert dm.target == "author"
        assert dm.num_classes == 4
        assert dm.dataset == "DBLP"
        assert dm.homogeneous is False

    def test_init_homogeneous_flag(self):
        """homogeneous=True is stored correctly."""
        dm = HGBBaseDataModule(homogeneous=True)
        assert dm.homogeneous is True


# ---------------------------------------------------------------------------
# TestHGBLinkPredBase -- constructor coverage
# ---------------------------------------------------------------------------


class TestHGBLinkPredBaseInit:
    def test_amazon_repr(self):
        dm = HGBAmazonLPDataModule(data_dir="/tmp/fake")
        assert repr(dm) == "HGBAmazon"

    def test_pubmed_repr(self):
        dm = HGBPubMedLPDataModule(data_dir="/tmp/fake")
        assert repr(dm) == "HGBPubMed"

    def test_youtube_repr(self):
        dm = HGBYouTubeLPDataModule(data_dir="/tmp/fake")
        assert repr(dm) == "HGBYouTube"

    def test_amazon_init_fields(self):
        dm = HGBAmazonLPDataModule(data_dir="/tmp/fake", homogeneous=True)
        assert dm.homogeneous is True
        assert "hgb_amazon" in dm.data_dir

    def test_pubmed_init_fields(self):
        dm = HGBPubMedLPDataModule(data_dir="/tmp/fake")
        assert "hgb_pubmed" in dm.data_dir

    def test_youtube_init_fields(self):
        dm = HGBYouTubeLPDataModule(data_dir="/tmp/fake")
        assert "hgb_youtube" in dm.data_dir

    def test_base_init_defaults(self):
        """HGBLinkPredBase.__init__ sets all attributes to None/defaults."""

        class _Concrete(HGBLinkPredBase):
            node_type_map = {0: "item"}
            edge_type_map = {}
            target = ("item", "rel", "item")
            rev_target = ("item", "rev_rel", "item")

        dm = _Concrete(data_dir="/tmp/fake", homogeneous=False)
        assert dm.train_data is None
        assert dm.val_data is None
        assert dm.test_data is None
        assert dm.num_node_types is None


# ---------------------------------------------------------------------------
# TestLoadHgbDataset -- missing mask error
# ---------------------------------------------------------------------------


class TestLoadHgbDatasetMissingMask:
    def test_missing_val_mask_raises_runtime_error(self):
        """_load_hgb_dataset raises RuntimeError when val_mask is missing."""
        from torch_geometric.data import HeteroData

        het = HeteroData()
        het["author"].x = torch.randn(10, 4)
        het["author"].num_nodes = 10
        het["paper"].x = torch.randn(5, 4)
        het["paper"].num_nodes = 5
        # Only provide train_mask; no val_mask (SubsetTrainSplit would add it,
        # but here we simulate the case where the flattened data is missing a mask)
        het["author"].train_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].train_mask[:5] = True
        # Deliberately omit val_mask and test_mask to trigger RuntimeError
        het["author", "cites", "paper"].edge_index = torch.randint(0, 5, (2, 10))

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            with pytest.raises(RuntimeError, match="val_mask|test_mask"):
                _load_hgb_dataset("dblp", root="/tmp/fake")


# ---------------------------------------------------------------------------
# TestHGBLinkPredBase -- parsing methods with synthetic files
# ---------------------------------------------------------------------------


class _MinimalLPDataModule(HGBLinkPredBase):
    """Minimal concrete subclass of HGBLinkPredBase for testing parsing methods."""

    node_type_map = {0: "item"}
    edge_type_map = {0: ("item", "also_viewed", "item")}
    target = ("item", "also_viewed", "item")
    rev_target = ("item", "rev_also_viewed", "item")
    test_edge_type_id = 0


class TestHGBLinkPredBaseParsing:
    @pytest.fixture
    def dm_with_files(self, tmp_path):
        """Set up a _MinimalLPDataModule with synthetic .dat files in tmp_path.

        Uses 10 nodes (ALL with features) and 20 edges for RandomLinkSplit.
        Every node has features to avoid KeyError in _build_hetero_data.
        """
        # Write node.dat: ALL 10 nodes have 2D features
        node_lines = [f"{i}\titem_{i}\t0\t{float(i)},{float(i + 1)}" for i in range(10)]
        node_dat = tmp_path / "node.dat"
        node_dat.write_text("\n".join(node_lines) + "\n", encoding="utf-8")

        # Write link.dat with 20 edges (enough for 70/15/15 split)
        link_lines = []
        for i in range(10):
            link_lines.append(f"{i}\t{(i + 1) % 10}\t0\t1.0")
            link_lines.append(f"{(i + 1) % 10}\t{i}\t0\t1.0")
        link_dat = tmp_path / "link.dat"
        link_dat.write_text("\n".join(link_lines) + "\n", encoding="utf-8")

        # Write link.dat.test with labels
        link_dat_test = tmp_path / "link.dat.test"
        link_dat_test.write_text("0\t2\t0\t1\n", encoding="utf-8")

        dm = _MinimalLPDataModule(data_dir=str(tmp_path), homogeneous=False)
        return dm

    def test_parse_nodes_returns_correct_structure(self, dm_with_files):
        """_parse_nodes() returns correct global_to_local, node_features, counts."""
        dm = dm_with_files
        g2l, feats, counts = dm._parse_nodes()

        # All 10 nodes should be in global_to_local
        for i in range(10):
            assert i in g2l, f"Node {i} should be in global_to_local"

        # All are type 0, local ids are sequential
        assert g2l[0] == (0, 0), f"Expected (0,0), got {g2l[0]}"
        assert g2l[1] == (0, 1), f"Expected (0,1), got {g2l[1]}"

        # All 10 nodes have features
        for i in range(10):
            assert i in feats, f"Node {i} should have features"

        # Node 0 has features [0.0, 1.0]
        assert feats[0] == [0.0, 1.0], f"Expected [0.0, 1.0], got {feats[0]}"
        assert counts[0] == 10, f"Expected 10 nodes of type 0, got {counts[0]}"

    def test_parse_links_returns_edge_lists(self, dm_with_files):
        """_parse_links() returns edge lists with correct src/dst."""
        dm = dm_with_files
        g2l, _, _ = dm._parse_nodes()
        edges = dm._parse_links("link.dat", g2l, has_label=False)

        assert 0 in edges, "Relation type 0 should have edges"
        assert len(edges[0]["src"]) == 20, (
            f"Expected 20 edges, got {len(edges[0]['src'])}"
        )

    def test_parse_links_nonexistent_file_returns_empty(self, dm_with_files):
        """_parse_links() returns {} when the file does not exist."""
        dm = dm_with_files
        g2l, _, _ = dm._parse_nodes()
        result = dm._parse_links("nonexistent.dat", g2l)
        assert result == {}

    def test_parse_links_with_labels(self, dm_with_files):
        """_parse_links() collects labels when has_label=True."""
        dm = dm_with_files
        g2l, _, _ = dm._parse_nodes()
        edges = dm._parse_links("link.dat.test", g2l, has_label=True)

        assert 0 in edges
        assert len(edges[0]["label"]) == 1
        assert edges[0]["label"][0] == 1

    def test_build_hetero_data_creates_features(self, dm_with_files):
        """_build_hetero_data() creates node features correctly."""
        dm = dm_with_files
        g2l, feats, counts = dm._parse_nodes()
        edges = dm._parse_links("link.dat", g2l, has_label=False)
        data = dm._build_hetero_data(g2l, feats, counts, edges)

        assert "item" in data.node_types
        # All nodes have features, so provided features are used
        assert data["item"].x is not None

    def test_build_hetero_data_no_features_uses_constant(self, tmp_path):
        """_build_hetero_data() uses constant features when no node has features."""
        node_dat = tmp_path / "node.dat"
        node_dat.write_text(
            "0\titem_a\t0\n1\titem_b\t0\n",
            encoding="utf-8",
        )
        dm = _MinimalLPDataModule(data_dir=str(tmp_path), homogeneous=False)
        g2l, feats, counts = dm._parse_nodes()
        data = dm._build_hetero_data(g2l, feats, counts, {})

        assert data["item"].x is not None
        # Constant features: shape (2, 128)
        assert data["item"].x.shape == (2, 128)

    def test_prepare_data_heterogeneous(self, dm_with_files):
        """prepare_data() populates metadata for heterogeneous mode."""
        dm = dm_with_files
        dm._download_and_extract = (  # type: ignore[method-assign]
            lambda: None
        )  # skip actual download
        dm.prepare_data()

        assert dm.train_data is not None
        assert dm.val_data is not None
        assert dm.test_data is not None
        assert dm.num_node_types is not None
        assert dm.num_edge_types is not None
        assert dm.node_type_names is not None

    def test_dataloaders_after_prepare(self, dm_with_files):
        """train/val/test dataloaders work after prepare_data()."""
        dm = dm_with_files
        dm._download_and_extract = lambda: None  # type: ignore[method-assign]
        dm.prepare_data()

        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None

    def test_prepare_data_homogeneous(self, tmp_path):
        """prepare_data() also works in homogeneous mode."""
        # 10 nodes with features, 20 edges (enough for RandomLinkSplit)
        node_lines = [f"{i}\titem_{i}\t0\t{float(i)},{float(i + 1)}" for i in range(10)]
        node_dat = tmp_path / "node.dat"
        node_dat.write_text("\n".join(node_lines) + "\n", encoding="utf-8")
        link_lines = [f"{i}\t{(i + 1) % 10}\t0\t1.0" for i in range(10)] + [
            f"{(i + 1) % 10}\t{i}\t0\t1.0" for i in range(10)
        ]
        link_dat = tmp_path / "link.dat"
        link_dat.write_text("\n".join(link_lines) + "\n", encoding="utf-8")

        dm = _MinimalLPDataModule(data_dir=str(tmp_path), homogeneous=True)
        dm._download_and_extract = lambda: None  # ty: ignore[invalid-assignment]
        dm.prepare_data()

        assert dm.train_data is not None
        assert dm.graph_size is not None
        assert dm.num_node_types is not None
        # In homogeneous mode, in_channels is an int
        assert isinstance(dm.in_channels, int)

    def test_parse_links_unknown_node_skipped(self, tmp_path):
        """_parse_links() skips edges where head or tail node is unknown."""
        node_dat = tmp_path / "node.dat"
        node_dat.write_text("0\titem_a\t0\n1\titem_b\t0\n", encoding="utf-8")

        # Edge references unknown node 99
        link_dat = tmp_path / "link.dat"
        link_dat.write_text("0\t99\t0\t1.0\n", encoding="utf-8")

        dm = _MinimalLPDataModule(data_dir=str(tmp_path), homogeneous=False)
        g2l, _, _ = dm._parse_nodes()
        edges = dm._parse_links("link.dat", g2l)
        # Edge referencing unknown node 99 should be skipped
        assert 0 not in edges or len(edges.get(0, {}).get("src", [])) == 0


# ---------------------------------------------------------------------------
# TestHGBBaseDataModule -- prepare_data (mocked)
# ---------------------------------------------------------------------------


class TestHGBBaseDataModulePrepareData:
    def test_prepare_data_heterogeneous(self):
        """HGBBaseDataModule.prepare_data() with mocked HGBDataset."""
        from torch_geometric.data import HeteroData

        het = HeteroData()
        het["author"].x = torch.randn(10, 4)
        het["author"].num_nodes = 10
        het["author"].train_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].train_mask[:5] = True
        het["author"].val_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].val_mask[5:7] = True
        het["author"].test_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].test_mask[8:] = True
        het["author", "cites", "paper"].edge_index = torch.randint(0, 10, (2, 20))
        het["paper"].x = torch.randn(15, 4)
        het["paper"].num_nodes = 15

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        dm = HGBBaseDataModule(
            target="author",
            num_classes=4,
            dataset="DBLP",
            homogeneous=False,
        )
        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            with patch("torch_geometric.data.lightning.LightningNodeData") as mock_lnd:
                mock_lnd.return_value.train_dataloader.return_value = MagicMock()
                mock_lnd.return_value.val_dataloader.return_value = MagicMock()
                mock_lnd.return_value.test_dataloader.return_value = MagicMock()
                dm.prepare_data()

        assert dm.in_channels is not None
        assert dm.num_nodes is not None
        assert dm.metadata is not None

    def test_prepare_data_homogeneous(self):
        """HGBBaseDataModule.prepare_data() in homogeneous mode."""
        from torch_geometric.data import HeteroData

        het = HeteroData()
        het["author"].x = torch.randn(10, 4)
        het["author"].num_nodes = 10
        het["author"].train_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].train_mask[:5] = True
        het["author"].val_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].val_mask[5:7] = True
        het["author"].test_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].test_mask[8:] = True
        het["author", "cites", "paper"].edge_index = torch.randint(0, 10, (2, 20))
        het["paper"].x = torch.randn(15, 4)
        het["paper"].num_nodes = 15

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        dm = HGBBaseDataModule(
            target="author",
            num_classes=4,
            dataset="DBLP",
            homogeneous=True,
        )
        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            with patch("torch_geometric.data.lightning.LightningNodeData") as mock_lnd:
                mock_lnd.return_value.train_dataloader.return_value = MagicMock()
                dm.prepare_data()

        assert isinstance(dm.in_channels, int), (
            "in_channels should be int in homogeneous mode"
        )
        assert dm.graph_size is not None
        assert dm.num_node_types is not None

    def _make_het_with_masks(self):
        """Helper: minimal HeteroData with all required masks."""
        from torch_geometric.data import HeteroData

        het = HeteroData()
        het["author"].x = torch.randn(10, 4)
        het["author"].num_nodes = 10
        het["author"].train_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].train_mask[:5] = True
        het["author"].val_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].val_mask[5:7] = True
        het["author"].test_mask = torch.zeros(10, dtype=torch.bool)
        het["author"].test_mask[8:] = True
        het["author", "cites", "paper"].edge_index = torch.randint(0, 10, (2, 20))
        het["paper"].x = torch.randn(15, 4)
        het["paper"].num_nodes = 15
        return het

    def test_dataloaders(self):
        """train/val/test dataloaders delegate to _pyg_datamodule."""
        het = self._make_het_with_masks()
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=het)

        train_dl = MagicMock()
        val_dl = MagicMock()
        test_dl = MagicMock()

        dm = HGBBaseDataModule(target="author", dataset="DBLP", homogeneous=False)
        with patch("exp.datasets.hgb.HGBDataset", return_value=mock_ds):
            with patch("torch_geometric.data.lightning.LightningNodeData") as mock_lnd:
                mock_lnd.return_value.train_dataloader.return_value = train_dl
                mock_lnd.return_value.val_dataloader.return_value = val_dl
                mock_lnd.return_value.test_dataloader.return_value = test_dl
                dm.prepare_data()

        assert dm.train_dataloader() is train_dl
        assert dm.val_dataloader() is val_dl
        assert dm.test_dataloader() is test_dl
