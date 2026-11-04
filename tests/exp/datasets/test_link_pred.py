# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/datasets/link_pred.py - LP DataModules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import HeteroData

from exp.data import DatasetInfo
from exp.datasets.info import LPDatasetInfo
from exp.datasets.link_pred import LastFMDataModule, MovieLensDataModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hetero(
    node_types: tuple[str, ...] = ("user", "artist"),
    edge_type: tuple[str, str, str] = ("user", "to", "artist"),
    n_src: int = 20,
    n_dst: int = 30,
    n_edges: int = 60,
) -> HeteroData:
    """Minimal HeteroData for mocking LP datasets."""
    src_t, rel, dst_t = edge_type
    data = HeteroData()
    data[src_t].x = torch.ones(n_src, 4)
    data[src_t].num_nodes = n_src
    data[dst_t].x = torch.ones(n_dst, 4)
    data[dst_t].num_nodes = n_dst
    src = torch.randint(0, n_src, (n_edges,))
    dst = torch.randint(0, n_dst, (n_edges,))
    data[src_t, rel, dst_t].edge_index = torch.stack([src, dst])
    return data


def _mock_pyg_dataset(hetero: HeteroData) -> MagicMock:
    ds = MagicMock()
    ds.__getitem__ = MagicMock(return_value=hetero)
    return ds


# ---------------------------------------------------------------------------
# TestLPDatasetInfo
# ---------------------------------------------------------------------------


class TestLPDatasetInfo:
    def test_construction_and_fields(self):
        info = LPDatasetInfo(
            name="lastfm",
            num_features=4,
            num_classes=1,
            num_splits=1,
            metric="roc_auc",
            split_type="lp_builtin",
            num_node_types=2,
            num_edge_types=1,
            node_type_names=("user", "artist"),
            edge_type_names=("user__to__artist",),
            target=("user", "to", "artist"),
        )
        assert info.metric == "roc_auc", f"Expected metric roc_auc, got {info.metric}"
        assert info.task == "binary", f"Expected binary task, got {info.task}"
        assert info.target == (
            "user",
            "to",
            "artist",
        ), f"Expected target ('user', 'to', 'artist'), got {info.target}"

    def test_is_dataset_info_subclass(self):
        info = LPDatasetInfo(
            name="lastfm",
            num_features=4,
            num_classes=1,
            num_splits=1,
            metric="roc_auc",
            split_type="lp_builtin",
        )
        assert isinstance(info, DatasetInfo), "info should be a DatasetInfo"

    def test_frozen(self):
        info = LPDatasetInfo(
            name="lastfm",
            num_features=4,
            num_classes=1,
            num_splits=1,
            metric="roc_auc",
            split_type="lp_builtin",
        )
        with pytest.raises((AttributeError, TypeError)):
            info.metric = "acc"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TestLastFMDataModule
# ---------------------------------------------------------------------------


class TestLastFMDataModule:
    @pytest.fixture
    def mock_data(self):
        return _make_hetero(
            node_types=("user", "artist"),
            edge_type=("user", "to", "artist"),
        )

    def test_target_and_rev_target(self):
        dm = LastFMDataModule(data_dir="/tmp/fake")
        assert dm.target == (
            "user",
            "to",
            "artist",
        ), f"Expected target ('user', 'to', 'artist'), got {dm.target}"
        assert dm.rev_target == (
            "artist",
            "to",
            "user",
        ), f"Expected rev_target ('artist', 'to', 'user'), got {dm.rev_target}"

    def test_prepare_data_sets_splits(self, mock_data):
        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=mock_data,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        assert dm.train_data is not None, (
            "Train data was not initialized in prepare_data"
        )
        assert dm.val_data is not None, "Val data was not initialized in prepare_data"
        assert dm.test_data is not None, "Test data was not initialized in prepare_data"

    def test_info_populated_after_prepare(self, mock_data):
        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=mock_data,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        assert dm.info is not None, "DatasetInfo was not populated"
        assert isinstance(dm.info, LPDatasetInfo), (
            "dm.info should be an instance of LPDatasetInfo"
        )
        assert isinstance(dm.info, DatasetInfo), (
            "dm.info should be an instance of DatasetInfo"
        )
        assert dm.info.metric == "roc_auc", f"Expected roc_auc, got {dm.info.metric}"
        assert dm.info.task == "binary", f"Expected binary task, got {dm.info.task}"
        assert dm.info.split_type == "lp_builtin", (
            f"Expected lp_builtin split_type, got {dm.info.split_type}"
        )
        assert dm.info.num_node_types >= 1, (
            f"Expected at least one node type, got {dm.info.num_node_types}"
        )
        assert dm.info.num_edge_types >= 1, (
            f"Expected at least one edge type, got {dm.info.num_edge_types}"
        )

    def test_heterogeneous_raises(self, mock_data):
        """prepare_data() must raise ValueError when homogeneous=False."""
        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=mock_data,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake", homogeneous=False)
            with pytest.raises(ValueError, match="homogeneous=True"):
                dm.prepare_data()

    def test_repr(self):
        assert repr(LastFMDataModule()) == "LastFM", (
            "Incorrect repr for LastFMDataModule"
        )

    def test_download_data_removes_presplit_keys(self):
        """Verify pre-split edge tensors are removed from the loaded data."""
        hetero = _make_hetero(
            edge_type=("user", "to", "artist"),
        )
        # Attach the tensors LastFM ships
        for key in (
            "train_neg_edge_index",
            "val_pos_edge_index",
            "val_neg_edge_index",
            "test_pos_edge_index",
            "test_neg_edge_index",
        ):
            hetero["user", "to", "artist"][key] = torch.empty(2, 5, dtype=torch.long)

        mock_ds = _mock_pyg_dataset(hetero)
        with patch("exp.datasets.link_pred.LastFM", return_value=mock_ds):
            dm = LastFMDataModule(data_dir="/tmp/fake")
            result = dm.download_data()

        edge_store = result["user", "to", "artist"]
        for key in (
            "train_neg_edge_index",
            "val_pos_edge_index",
            "val_neg_edge_index",
            "test_pos_edge_index",
            "test_neg_edge_index",
        ):
            assert key not in edge_store, f"Key {key} was not removed from edge_store"


# ---------------------------------------------------------------------------
# TestMovieLensDataModule
# ---------------------------------------------------------------------------


class TestMovieLensDataModule:
    @pytest.fixture
    def mock_data(self):
        return _make_hetero(
            node_types=("user", "movie"),
            edge_type=("user", "rates", "movie"),
        )

    def test_target_and_rev_target(self):
        dm = MovieLensDataModule(data_dir="/tmp/fake")
        assert dm.target == (
            "user",
            "rates",
            "movie",
        ), f"Expected target ('user', 'rates', 'movie'), got {dm.target}"
        assert dm.rev_target == (
            "movie",
            "rev_rates",
            "user",
        ), f"Expected rev_target ('movie', 'rev_rates', 'user'), got {dm.rev_target}"

    def test_prepare_data_sets_splits(self, mock_data):
        with patch(
            "exp.datasets.link_pred.MovieLensDataModule.download_data",
            return_value=mock_data,
        ):
            dm = MovieLensDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        assert dm.train_data is not None, (
            "Train data was not initialized in prepare_data (MovieLens)"
        )
        assert dm.val_data is not None, (
            "Val data was not initialized in prepare_data (MovieLens)"
        )
        assert dm.test_data is not None, (
            "Test data was not initialized in prepare_data (MovieLens)"
        )

    def test_info_populated_after_prepare(self, mock_data):
        with patch(
            "exp.datasets.link_pred.MovieLensDataModule.download_data",
            return_value=mock_data,
        ):
            dm = MovieLensDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        assert dm.info is not None, "DatasetInfo was not populated (MovieLens)"
        assert isinstance(dm.info, LPDatasetInfo), (
            "dm.info should be LPDatasetInfo (MovieLens)"
        )
        assert dm.info.metric == "roc_auc", f"Expected roc_auc, got {dm.info.metric}"
        assert dm.info.target == (
            "user",
            "rates",
            "movie",
        ), f"Expected target ('user', 'rates', 'movie'), got {dm.info.target}"

    def test_repr(self):
        assert repr(MovieLensDataModule()) == "MovieLens", (
            f"Expected repr MovieLens, got {repr(MovieLensDataModule())}"
        )

    def test_download_data_removes_edge_labels(self):
        """Verify pre-attached rating edge_labels are stripped."""
        hetero = _make_hetero(
            node_types=("user", "movie"),
            edge_type=("user", "rates", "movie"),
        )
        hetero["user", "rates", "movie"].edge_label = torch.rand(60)
        hetero["movie", "rev_rates", "user"].edge_index = torch.empty(
            2, 0, dtype=torch.long
        )
        hetero["movie", "rev_rates", "user"].edge_label = torch.rand(0)

        mock_ds = _mock_pyg_dataset(hetero)
        with patch("exp.datasets.link_pred._MovieLens", return_value=mock_ds):
            dm = MovieLensDataModule(data_dir="/tmp/fake")
            result = dm.download_data()

        assert "edge_label" not in result["user", "rates", "movie"], (
            "edge_label was not removed from result"
        )

    def test_homogeneous_flag_accepted(self, mock_data):
        with patch(
            "exp.datasets.link_pred.MovieLensDataModule.download_data",
            return_value=mock_data,
        ):
            dm = MovieLensDataModule(data_dir="/tmp/fake", homogeneous=True)
            dm.prepare_data()

        assert dm.train_data is not None, (
            "Train data was not initialized when homogeneous=True"
        )


# ---------------------------------------------------------------------------
# TestLinkPredBase -- missing branches
# ---------------------------------------------------------------------------


class TestLinkPredBaseMissingBranches:
    def test_degree_features_set_after_homogeneous_convert(self):
        """Lines 172-184: node_feature_type='degree' sets log-degree features."""
        mock_data = _make_hetero(
            node_types=("user", "artist"),
            edge_type=("user", "to", "artist"),
        )
        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=mock_data,
        ):
            dm = LastFMDataModule(
                data_dir="/tmp/fake",
                homogeneous=True,
                node_feature_type="degree",
            )
            dm.prepare_data()

        assert dm.train_data is not None
        # After homogeneous conversion with degree features, x should have 2 cols
        # (log(out_deg+1), log(in_deg+1))
        assert dm.info is not None
        assert dm.info.num_features == 2, (
            f"Expected 2 degree features, got {dm.info.num_features}"
        )
