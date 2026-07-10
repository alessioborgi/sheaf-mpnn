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

    # -- Planetoid loader (cora / citeseer) --

    def test_planetoid_cora_returns_correct_info(self):
        data = _synthetic_data(num_nodes=20, num_features=5, num_classes=3)
        with patch("exp.data.Planetoid", return_value=_mock_ds(data)):
            result_data, info = load_dataset("cora", root="/tmp/fake")
        assert info.name == "cora"
        assert info.num_features == 5
        assert info.num_classes == 3
        assert info.num_splits == 10
        assert info.metric == "acc"
        assert info.split_type == "npz_file"

    def test_planetoid_citeseer_returns_correct_info(self):
        data = _synthetic_data(num_nodes=15, num_features=6, num_classes=4)
        with patch("exp.data.Planetoid", return_value=_mock_ds(data)):
            result_data, info = load_dataset("citeseer", root="/tmp/fake")
        assert info.name == "citeseer"
        assert info.split_type == "npz_file"

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

    # -- HeterophilousGraphDataset loader --

    def test_heterophilous_minesweeper_roc_auc(self):
        data = _synthetic_multicol_data(num_nodes=30, num_features=7, num_classes=2)
        with patch("exp.data.HeterophilousGraphDataset", return_value=_mock_ds(data)):
            result_data, info = load_dataset("minesweeper", root="/tmp/fake")
        assert info.metric == "roc_auc"
        assert info.split_type == "pyg_mask"
        assert info.num_splits == 10

    def test_heterophilous_amazon_ratings_acc(self):
        data = _synthetic_multicol_data(num_nodes=30, num_features=9, num_classes=5)
        with patch("exp.data.HeterophilousGraphDataset", return_value=_mock_ds(data)):
            result_data, info = load_dataset("amazon_ratings", root="/tmp/fake")
        assert info.metric == "acc"
        assert info.split_type == "pyg_mask"

    def test_heterophilous_tolokers_roc_auc(self):
        data = _synthetic_multicol_data(num_nodes=20, num_features=5, num_classes=2)
        with patch("exp.data.HeterophilousGraphDataset", return_value=_mock_ds(data)):
            result_data, info = load_dataset("tolokers", root="/tmp/fake")
        assert info.metric == "roc_auc"

    # -- WikipediaNetwork loader (chameleon / squirrel) --

    def test_wiki_chameleon_returns_correct_info(self):
        data = _synthetic_multicol_data(num_nodes=20, num_features=5, num_classes=5)
        with patch("exp.data.WikipediaNetwork", return_value=_mock_ds(data)):
            result_data, info = load_dataset("chameleon", root="/tmp/fake")
        assert info.name == "chameleon"
        assert info.split_type == "npz_file"

    # -- Returned Data object --

    def test_returned_data_has_correct_num_features(self):
        data = _synthetic_data(num_nodes=20, num_features=8, num_classes=3)
        with patch("exp.data.Planetoid", return_value=_mock_ds(data)):
            result_data, info = load_dataset("cora", root="/tmp/fake")
        assert isinstance(result_data, Data) and result_data.x is not None
        assert result_data.x.shape[1] == 8

    def test_num_classes_derived_from_max_label(self):
        data = _synthetic_data(num_nodes=20, num_features=5, num_classes=4)
        with patch("exp.data.Planetoid", return_value=_mock_ds(data)):
            result_data, info = load_dataset("cora", root="/tmp/fake")
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
        name = "chameleon_filtered"
        root = str(tmp_path)
        # PyG's InMemoryDataset adds the dataset
        # name to the root if it's not already there
        # but FilteredWikipediaDataset uses self.raw_dir which is root/raw
        raw_dir = os.path.join(root, name, "raw")
        os.makedirs(raw_dir)
        np.savez(os.path.join(raw_dir, f"{name}.npz"), **mock_npz)

        with patch("exp.data.urllib.request.urlretrieve"):
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
