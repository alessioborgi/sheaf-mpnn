# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/data.py -- dataset loading, DatasetInfo, and SheafDataModule."""

from __future__ import annotations

import os
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from exp.data import (
    _LOADER,
    NPZ_SPLIT_DATASETS,
    ROC_AUC_DATASETS,
    DatasetInfo,
    _canonical,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_data(num_nodes: int, num_features: int, num_classes: int) -> Data:
    """Minimal graph Data object for mocking dataset loads."""
    return Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
        y=torch.randint(0, num_classes, (num_nodes,)),
        train_mask=torch.zeros(num_nodes, dtype=torch.bool),
        val_mask=torch.zeros(num_nodes, dtype=torch.bool),
        test_mask=torch.zeros(num_nodes, dtype=torch.bool),
    )


def _synthetic_multicol_data(
    num_nodes: int, num_features: int, num_classes: int, n_splits: int = 10
) -> Data:
    """Data object with multi-column masks (PyG heterophilous convention)."""
    return Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
        y=torch.randint(0, num_classes, (num_nodes,)),
        train_mask=torch.zeros(num_nodes, n_splits, dtype=torch.bool),
        val_mask=torch.zeros(num_nodes, n_splits, dtype=torch.bool),
        test_mask=torch.zeros(num_nodes, n_splits, dtype=torch.bool),
    )


def _mock_ds(data: Data) -> MagicMock:
    ds = MagicMock()
    ds.__getitem__ = MagicMock(return_value=data)
    return ds


# ---------------------------------------------------------------------------
# _canonical
# ---------------------------------------------------------------------------


class TestCanonical:
    def test_lowercases_input(self):
        assert _canonical("CORA") == "cora"

    def test_replaces_hyphens_with_underscores(self):
        assert _canonical("chameleon-filtered") == "chameleon_filtered"

    def test_strips_whitespace(self):
        assert _canonical("  texas  ") == "texas"

    def test_alias_fil_resolves_to_film(self):
        assert _canonical("fil") == "film"

    def test_all_loader_keys_are_canonical(self):
        for name in _LOADER:
            assert _canonical(name) == name

    def test_unknown_name_passes_through(self):
        assert _canonical("totally_unknown") == "totally_unknown"


# ---------------------------------------------------------------------------
# DatasetInfo
# ---------------------------------------------------------------------------


class TestDatasetInfo:
    def test_construction_and_fields(self):
        info = DatasetInfo(
            name="cora",
            num_features=1433,
            num_classes=7,
            num_splits=10,
            metric="acc",
            split_type="npz_file",
        )
        assert info.name == "cora"
        assert info.num_features == 1433
        assert info.num_classes == 7
        assert info.num_splits == 10
        assert info.metric == "acc"
        assert info.split_type == "npz_file"

    def test_frozen_raises_on_assignment(self):
        info = DatasetInfo("cora", 1433, 7, 10, "acc", "npz_file")
        with pytest.raises((TypeError, AttributeError)):
            info.name = "texas"  # ty: ignore[invalid-assignment]

    def test_roc_auc_metric_field(self):
        info = DatasetInfo("minesweeper", 5, 2, 10, "roc_auc", "pyg_mask")
        assert info.metric == "roc_auc"


# ---------------------------------------------------------------------------
# Dataset metadata constants
# ---------------------------------------------------------------------------


class TestROCAUCDatasets:
    def test_roc_auc_datasets_subset_of_loader(self):
        assert ROC_AUC_DATASETS.issubset(set(_LOADER.keys()))

    @pytest.mark.parametrize("name", ["minesweeper", "tolokers", "questions"])
    def test_known_roc_auc_datasets_present(self, name):
        assert name in ROC_AUC_DATASETS

    @pytest.mark.parametrize("name", ["cora", "citeseer", "texas", "cornell"])
    def test_acc_datasets_not_in_roc_auc(self, name):
        assert name not in ROC_AUC_DATASETS


class TestNPZSplitDatasets:
    def test_npz_split_datasets_subset_of_loader(self):
        assert NPZ_SPLIT_DATASETS.issubset(set(_LOADER.keys()))

    @pytest.mark.parametrize(
        "name",
        ["cora", "citeseer", "chameleon", "squirrel", "cornell", "texas", "film"],
    )
    def test_known_npz_datasets_present(self, name):
        assert name in NPZ_SPLIT_DATASETS

    @pytest.mark.parametrize("name", ["amazon_ratings", "minesweeper", "roman_empire"])
    def test_heterophilous_not_in_npz(self, name):
        assert name not in NPZ_SPLIT_DATASETS


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_unknown_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("totally_nonexistent_xyz")

    def test_unknown_with_helpful_message(self):
        with pytest.raises(ValueError, match="Supported"):
            load_dataset("garbage_name")

    # -- Directed citation override (cora / citeseer / pubmed) --
    # These names are canonically loaded from the DSNN *directed* citation
    # source, not PyG's Planetoid; the undirected variant is the default and the
    # directed graph is selected with the ``--directed`` flag.

    def test_cora_routes_to_directed_citation(self):
        data = _synthetic_multicol_data(num_nodes=20, num_features=5, num_classes=3)
        info = DatasetInfo("cora", 5, 3, 10, "acc", "pyg_mask")
        with (
            patch("exp.datasets.directed.is_directed_citation", return_value=True),
            patch(
                "exp.datasets.directed.load_directed_citation",
                return_value=(data, info),
            ),
        ):
            _result_data, out = load_dataset("cora", root="/tmp/fake")
        assert out.name == "cora"
        assert out.num_features == 5
        assert out.num_classes == 3
        assert out.split_type == "pyg_mask"

    def test_citeseer_routes_to_directed_citation(self):
        data = _synthetic_multicol_data(num_nodes=15, num_features=6, num_classes=4)
        info = DatasetInfo("citeseer", 6, 4, 10, "acc", "pyg_mask")
        with (
            patch("exp.datasets.directed.is_directed_citation", return_value=True),
            patch(
                "exp.datasets.directed.load_directed_citation",
                return_value=(data, info),
            ),
        ):
            _result_data, out = load_dataset("citeseer", root="/tmp/fake")
        assert out.name == "citeseer"
        assert out.split_type == "pyg_mask"

    # -- WebKB loader (cornell / texas) --

    def test_webkb_texas_returns_correct_info(self):
        data = _synthetic_data(num_nodes=15, num_features=4, num_classes=5)
        with patch("exp.data.WebKB", return_value=_mock_ds(data)):
            result_data, info = load_dataset("texas", root="/tmp/fake")
        assert info.name == "texas"
        assert info.split_type == "npz_file"
        assert info.metric == "acc"

    def test_webkb_cornell_returns_correct_info(self):
        data = _synthetic_data(num_nodes=18, num_features=3, num_classes=5)
        with patch("exp.data.WebKB", return_value=_mock_ds(data)):
            result_data, info = load_dataset("cornell", root="/tmp/fake")
        assert info.name == "cornell"
        assert info.num_splits == 10

    # -- Actor loader (film) --

    def test_actor_film_returns_correct_info(self):
        data = _synthetic_data(num_nodes=25, num_features=7, num_classes=5)
        with patch("exp.data.Actor", return_value=_mock_ds(data)):
            result_data, info = load_dataset("film", root="/tmp/fake")
        assert info.name == "film"
        assert info.split_type == "npz_file"
        assert info.num_splits == 10

    def test_alias_fil_accepted(self):
        data = _synthetic_data(num_nodes=25, num_features=7, num_classes=5)
        with patch("exp.data.Actor", return_value=_mock_ds(data)):
            result_data, info = load_dataset("fil", root="/tmp/fake")
        assert info.name == "film"

    # -- Directed Platonov override (heterophilous .npz) --
    # The Platonov heterophilous names are canonically loaded from the DSNN
    # *directed* .npz source, not PyG's HeterophilousGraphDataset.

    def test_minesweeper_routes_to_directed_npz(self):
        data = _synthetic_multicol_data(num_nodes=30, num_features=7, num_classes=2)
        info = DatasetInfo("minesweeper", 7, 2, 10, "roc_auc", "pyg_mask")
        with (
            patch("exp.datasets.directed.is_directed_npz", return_value=True),
            patch("exp.datasets.directed.load_directed_npz", return_value=(data, info)),
        ):
            _result_data, out = load_dataset("minesweeper", root="/tmp/fake")
        assert out.metric == "roc_auc"
        assert out.split_type == "pyg_mask"
        assert out.num_splits == 10

    def test_amazon_ratings_routes_to_directed_npz(self):
        data = _synthetic_multicol_data(num_nodes=30, num_features=9, num_classes=5)
        info = DatasetInfo("amazon_ratings", 9, 5, 10, "acc", "pyg_mask")
        with (
            patch("exp.datasets.directed.is_directed_npz", return_value=True),
            patch("exp.datasets.directed.load_directed_npz", return_value=(data, info)),
        ):
            _result_data, out = load_dataset("amazon_ratings", root="/tmp/fake")
        assert out.metric == "acc"
        assert out.split_type == "pyg_mask"

    # -- WikipediaNetwork loader (chameleon / squirrel) --

    def test_wiki_chameleon_returns_correct_info(self):
        data = _synthetic_multicol_data(num_nodes=20, num_features=5, num_classes=5)
        with patch("exp.data.WikipediaNetwork", return_value=_mock_ds(data)):
            result_data, info = load_dataset("chameleon", root="/tmp/fake")
        assert info.name == "chameleon"
        assert info.split_type == "npz_file"

    # -- OGB loader (ogbn_arxiv) --

    @staticmethod
    def _mock_ogb_ds(num_nodes: int = 30) -> tuple[MagicMock, Data]:
        # OGB convention: y is [N, 1], no masks; get_idx_split() returns the
        # official single split as index tensors.
        data = Data(
            x=torch.randn(num_nodes, 6),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
            y=torch.randint(0, 4, (num_nodes, 1)),
        )
        ds = _mock_ds(data)
        idx = torch.randperm(num_nodes)
        ds.get_idx_split = MagicMock(
            return_value={
                "train": idx[: num_nodes // 2],
                "valid": idx[num_nodes // 2 : num_nodes * 3 // 4],
                "test": idx[num_nodes * 3 // 4 :],
            }
        )
        return ds, data

    def test_ogbn_arxiv_returns_correct_info(self):
        ds, _ = self._mock_ogb_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, info = load_dataset("ogbn-arxiv", root="/tmp/fake")
        assert info.name == "ogbn_arxiv"
        assert info.metric == "acc"
        assert info.split_type == "pyg_mask"
        assert info.num_splits == 1
        assert info.num_classes == 4

    def test_ogbn_arxiv_stamps_official_split_as_masks(self):
        ds, _ = self._mock_ogb_ds(num_nodes=40)
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, info = load_dataset("ogbn_arxiv", root="/tmp/fake")
        for mask_name, size in (
            ("train_mask", 20),
            ("val_mask", 10),
            ("test_mask", 10),
        ):
            mask = result_data[mask_name]
            assert mask.dtype == torch.bool and mask.dim() == 1
            assert int(mask.sum()) == size
        # The three masks partition the official split without overlap.
        overlap = result_data.train_mask & (
            result_data.val_mask | result_data.test_mask
        )
        assert not bool(overlap.any())

    def test_ogbn_arxiv_flattens_labels(self):
        ds, _ = self._mock_ogb_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, _ = load_dataset("ogbn_arxiv", root="/tmp/fake")
        assert isinstance(result_data.y, torch.Tensor)
        assert result_data.y.dim() == 1
        assert result_data.y.dtype == torch.long

    def test_ogbn_arxiv_symmetrized_by_default(self):
        from torch_geometric.utils import is_undirected  # noqa: PLC0415

        ds, _ = self._mock_ogb_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, _ = load_dataset("ogbn_arxiv", root="/tmp/fake")
        assert result_data.edge_index is not None
        assert is_undirected(result_data.edge_index)

    def test_ogbn_products_uses_same_ogb_path(self):
        ds, _ = self._mock_ogb_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, info = load_dataset("ogbn-products", root="/tmp/fake")
        assert info.name == "ogbn_products"
        assert info.metric == "acc"
        assert info.task == "multiclass"
        assert info.num_splits == 1
        assert isinstance(result_data.y, torch.Tensor)
        assert result_data.y.dim() == 1

    # -- OGB loader (ogbn_proteins: featureless, multilabel) --

    @staticmethod
    def _mock_proteins_ds(
        num_nodes: int = 30, num_tasks: int = 7, edge_dim: int = 8
    ) -> MagicMock:
        # ogbn-proteins convention: no x, [E, 8] edge features, [N, T] binary y.
        num_edges = num_nodes * 4
        data = Data(
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.rand(num_edges, edge_dim),
            y=torch.randint(0, 2, (num_nodes, num_tasks)),
            num_nodes=num_nodes,
        )
        ds = _mock_ds(data)
        idx = torch.randperm(num_nodes)
        ds.get_idx_split = MagicMock(
            return_value={
                "train": idx[: num_nodes // 2],
                "valid": idx[num_nodes // 2 : num_nodes * 3 // 4],
                "test": idx[num_nodes * 3 // 4 :],
            }
        )
        return ds

    def test_ogbn_proteins_returns_multilabel_info(self):
        ds = self._mock_proteins_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            _result_data, info = load_dataset("ogbn-proteins", root="/tmp/fake")
        assert info.name == "ogbn_proteins"
        assert info.metric == "roc_auc"
        assert info.task == "multilabel"
        assert info.num_classes == 7
        assert info.num_features == 8
        assert info.num_splits == 1

    def test_ogbn_proteins_builds_node_features_from_edge_means(self):
        ds = self._mock_proteins_ds(num_nodes=25)
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, _ = load_dataset("ogbn_proteins", root="/tmp/fake")
        assert result_data.x is not None
        assert result_data.x.shape == (25, 8)
        # Edge features are in [0, 1), so their means must be too.
        assert float(result_data.x.min()) >= 0.0
        assert float(result_data.x.max()) < 1.0
        assert result_data.edge_attr is None

    def test_ogbn_proteins_keeps_2d_float_labels(self):
        ds = self._mock_proteins_ds()
        with patch("ogb.nodeproppred.PygNodePropPredDataset", return_value=ds):
            result_data, _ = load_dataset("ogbn_proteins", root="/tmp/fake")
        assert isinstance(result_data.y, torch.Tensor)
        assert result_data.y.shape == (30, 7)
        assert result_data.y.dtype == torch.float32

    # -- LINKX loader (pokec) --

    @staticmethod
    def _fake_pokec_files(root, num_nodes: int = 20, num_splits: int = 5):
        # Mirror the LINKX release layout: pokec.mat plus pokec-splits.npy
        # holding per-fold train/valid/test index dicts over labeled nodes.
        import scipy.io  # noqa: PLC0415

        raw = os.path.join(root, "pokec", "raw")
        os.makedirs(raw, exist_ok=True)
        rng = np.random.default_rng(0)
        label = rng.integers(0, 2, num_nodes)
        label[:2] = -1  # unlabeled nodes, excluded from every split
        scipy.io.savemat(
            os.path.join(raw, "pokec.mat"),
            {
                "edge_index": rng.integers(0, num_nodes, (2, num_nodes * 3)),
                "node_feat": rng.random((num_nodes, 4)),
                "label": label,
            },
        )
        labeled = np.nonzero(label >= 0)[0]
        splits = []
        for i in range(num_splits):
            perm = np.random.default_rng(i).permutation(labeled)
            n = len(perm)
            splits.append(
                {
                    "train": perm[: n // 2],
                    "valid": perm[n // 2 : 3 * n // 4],
                    "test": perm[3 * n // 4 :],
                }
            )
        np.save(
            os.path.join(raw, "pokec-splits.npy"),
            np.asarray(splits, dtype=object),
            allow_pickle=True,
        )

    def test_pokec_loads_official_splits_as_mask_columns(self, tmp_path):
        root = str(tmp_path)
        self._fake_pokec_files(root)
        result_data, info = load_dataset("pokec", root=root)
        assert info.name == "pokec"
        assert info.metric == "acc"
        assert info.task == "multiclass"
        assert info.num_classes == 2
        assert info.num_splits == 5
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = result_data[mask_name]
            assert mask.shape == (20, 5) and mask.dtype == torch.bool
        # Unlabeled nodes (-1) never appear in any split column.
        assert isinstance(result_data.y, torch.Tensor)
        unlabeled = result_data.y < 0
        assert not bool(
            (result_data.train_mask | result_data.val_mask | result_data.test_mask)[
                unlabeled
            ].any()
        )

    def test_pokec_caches_processed_data(self, tmp_path):
        root = str(tmp_path)
        self._fake_pokec_files(root)
        load_dataset("pokec", root=root)
        assert os.path.isfile(os.path.join(root, "pokec", "processed", "pokec.pt"))

    # -- Returned Data object --

    def test_returned_data_has_correct_num_features(self):
        data = _synthetic_data(num_nodes=20, num_features=8, num_classes=3)
        with patch("exp.data.WebKB", return_value=_mock_ds(data)):
            result_data, info = load_dataset("texas", root="/tmp/fake")
        assert isinstance(result_data, Data) and result_data.x is not None
        assert result_data.x.shape[1] == 8

    def test_num_classes_derived_from_max_label(self):
        data = _synthetic_data(num_nodes=20, num_features=5, num_classes=4)
        with patch("exp.data.WebKB", return_value=_mock_ds(data)):
            result_data, info = load_dataset("texas", root="/tmp/fake")
        assert isinstance(data.y, torch.Tensor)
        assert info.num_classes == int(data.y.max().item()) + 1


# ---------------------------------------------------------------------------
# SheafDataModule
# ---------------------------------------------------------------------------


class TestSheafDataModule:
    @pytest.fixture
    def setup_data(self):
        data = _synthetic_data(num_nodes=20, num_features=5, num_classes=3)
        data.train_mask[:12] = True
        data.val_mask[12:16] = True
        data.test_mask[16:] = True
        info = DatasetInfo(
            name="cora",
            num_features=5,
            num_classes=3,
            num_splits=10,
            metric="acc",
            split_type="npz_file",
        )
        return data, info

    def test_info_raises_runtime_error_before_setup(self):
        from exp.data import SheafDataModule

        dm = SheafDataModule("cora", root="/tmp/fake", fold=0)
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.info

    def test_setup_populates_info(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule("cora", root="/tmp/fake", fold=0)
            dm.setup()
        assert dm.info.name == "cora"
        assert dm.info.num_features == 5

    def test_setup_not_repeated_on_second_call(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)) as mock_load,
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule("cora", root="/tmp/fake", fold=0)
            dm.setup()
            dm.setup()  # second call
        mock_load.assert_called_once()

    def test_train_dataloader_returns_batch(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule("cora", root="/tmp/fake", fold=0)
            dm.setup()
            batch = next(iter(dm.train_dataloader()))
        assert batch.num_nodes == 20

    def test_all_dataloaders_return_same_graph(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule("cora", root="/tmp/fake", fold=0)
            dm.setup()
            train_batch = next(iter(dm.train_dataloader()))
            val_batch = next(iter(dm.val_dataloader()))
            test_batch = next(iter(dm.test_dataloader()))
        # Full-graph transductive: all loaders serve the same graph
        assert train_batch.num_nodes == val_batch.num_nodes == test_batch.num_nodes

    def test_fold_is_forwarded_to_apply_split(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data) as mock_split,
        ):
            dm = SheafDataModule("cora", root="/tmp/fake", fold=7)
            dm.setup()
        _, _, fold_arg = mock_split.call_args.args
        assert fold_arg == 7

    def test_properties_raise_before_setup(self):
        from exp.data import SheafDataModule

        dm = SheafDataModule("cora", root="/tmp/fake")
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_nodes
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.num_edges
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.homophily
        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.split_sizes

    def test_properties_after_setup(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        data.edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]]
        )  # 2 undirected edges
        data.y = torch.tensor([0, 0, 1])  # edge (0,1) is homophilic, (1,2) is not
        data.train_mask = torch.tensor([True, False, False])
        data.val_mask = torch.tensor([False, True, False])
        data.test_mask = torch.tensor([False, False, True])

        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule("cora", root="/tmp/fake")
            dm.setup()
            assert dm.num_nodes == data.x.size(0)
            assert dm.num_edges == 2
            assert dm.homophily == 0.5
            assert dm.split_sizes == (1, 1, 1)

    def test_hardware_propagation(self, setup_data):
        from exp.data import SheafDataModule

        data, info = setup_data
        with (
            patch("exp.data.load_dataset", return_value=(data, info)),
            patch("exp.splits.apply_split", return_value=data),
        ):
            dm = SheafDataModule(
                "cora",
                root="/tmp/fake",
                batch_size=4,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
            )
            dm.setup()
            dl = dm.train_dataloader()
            assert dl.batch_size == 4
            assert dl.num_workers == 2
            assert dl.pin_memory is True
            assert dl.persistent_workers is True


# ---------------------------------------------------------------------------
# FilteredWikipediaDataset & load_dataset(filtered_wiki)
# ---------------------------------------------------------------------------


class TestFilteredWikipediaDataset:
    @pytest.fixture
    def mock_npz(self):
        rng = np.random.default_rng()
        return {
            "node_features": rng.standard_normal((10, 4)).astype(np.float32),
            "node_labels": rng.integers(0, 2, size=10).astype(np.int64),
            "edges": np.array([[0, 1], [1, 2], [2, 3]]).astype(np.int64),
            "train_masks": np.zeros((10, 10), dtype=bool),
            "val_masks": np.zeros((10, 10), dtype=bool),
            "test_masks": np.zeros((10, 10), dtype=bool),
        }

    def test_process_and_load(self, tmp_path, mock_npz):
        from exp.data import FilteredWikipediaDataset

        name = "chameleon_filtered"
        root = str(tmp_path)
        raw_dir = os.path.join(root, "raw")
        os.makedirs(raw_dir)
        np.savez(os.path.join(raw_dir, f"{name}.npz"), **mock_npz)

        # Mock download to avoid network hits
        with patch.object(FilteredWikipediaDataset, "download", return_value=None):
            ds = FilteredWikipediaDataset(root, name)

        assert len(ds) == 1
        data = cast(Data, ds[0])
        assert data.x is not None
        assert data.edge_index is not None
        assert data.train_mask is not None
        assert data.x.shape == (10, 4)
        assert data.edge_index.shape[0] == 2
        # train_mask should be (N, 10)
        assert data.train_mask.shape == (10, 10)

    def test_load_dataset_filtered_wiki(self, tmp_path, mock_npz):
        # chameleon_filtered is canonically served by the DSNN directed source;
        # a file dropped under <root>/directed takes precedence over downloads.
        name = "chameleon_filtered"
        root = str(tmp_path)
        directed_dir = os.path.join(root, "directed")
        os.makedirs(directed_dir)
        np.savez(os.path.join(directed_dir, f"{name}.npz"), **mock_npz)

        data, info = load_dataset(name, root=root)

        assert info.name == name
        assert info.split_type == "pyg_mask"
        assert info.num_splits == 10

    def test_download(self, tmp_path):
        from exp.data import FilteredWikipediaDataset

        name = "chameleon_filtered"
        root = str(tmp_path)
        with (
            patch("exp.data.urllib.request.urlretrieve") as mock_retrieve,
            patch.object(FilteredWikipediaDataset, "_process"),
            patch("torch.load", return_value=(None, None)),
        ):
            ds = FilteredWikipediaDataset(root, name)
            ds.download()

        mock_retrieve.assert_called()
        args, _ = mock_retrieve.call_args
        assert args[0].endswith("chameleon_filtered.npz")
        assert args[1].endswith(os.path.join("raw", "chameleon_filtered.npz"))
