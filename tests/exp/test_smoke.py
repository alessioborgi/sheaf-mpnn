# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end smoke tests for all three training pipelines.

Each test runs a full Lightning Trainer (1 epoch, CPU, no checkpointing) with
synthetic / mocked data to verify the DataModule -> LightningModule -> Trainer
chain works without errors.  No real dataset downloads occur.

Pipelines covered
-----------------
* HGB node classification : HetSheafDataModule + SheafLightningModule
* GC (homogeneous)        : GCSheafDataModule  + GCLightningModule
* GC (heterogeneous)      : GCHetSheafDataModule + GCLightningModule
* LP DataModules          : LastFMDataModule / MovieLensDataModule (prepare + batch)
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from exp.config import Config, ModelConfig, OptimConfig, RegConfig

# ---------------------------------------------------------------------------
# Shared config factory
# ---------------------------------------------------------------------------


def _cfg(num_features: int = 4, num_classes: int = 3) -> tuple[Config, int, int]:
    """Return a tiny Config plus a matching DatasetInfo-like stub."""
    cfg = Config(
        model=ModelConfig(
            variant="general",
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            alpha=1.0,
        ),
        reg=RegConfig(input_dropout=0.0, dropout=0.0),
        optim=OptimConfig(lr=0.01, epochs=1, early_stopping=999),
    )
    return cfg, num_features, num_classes


# ---------------------------------------------------------------------------
# HGB smoke test
# ---------------------------------------------------------------------------


def _make_het_data(n: int = 30, f: int = 4, num_nt: int = 2) -> Data:
    n_edges = n * 2
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:12] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[12:18] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[18:] = True
    return Data(
        x=torch.randn(n, f),
        edge_index=torch.randint(0, n, (2, n_edges)),
        y=torch.randint(0, 3, (n,)),
        node_type=torch.randint(0, num_nt, (n,)),
        edge_type=torch.randint(0, 3, (n_edges,)),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _make_het_info(f: int = 4):
    from exp.datasets.info import HetDatasetInfo

    return HetDatasetInfo(
        name="dblp",
        num_features=f,
        num_classes=3,
        num_splits=1,
        metric="acc",
        split_type="hgb_builtin",
        num_node_types=2,
        num_edge_types=3,
        node_type_names=("author", "paper"),
        edge_type_names=("cites",),
        task="multiclass",
    )


class TestHGBSmoke:
    def test_nsd_one_epoch(self):
        from lightning import Trainer

        from exp.datasets.hgb import HetSheafDataModule
        from exp.module import SheafLightningModule

        data = _make_het_data()
        info = _make_het_info()
        cfg, num_features, num_classes = _cfg(num_features=4, num_classes=3)

        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule("dblp", root="/tmp/fake")
            module = SheafLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_acc" in result, f"Expected test_acc in result, got {result.keys()}"
        assert 0.0 <= result["test_acc"] <= 1.0, (
            f"Expected test_acc between 0 and 1, got {result['test_acc']}"
        )

    @pytest.mark.parametrize("dataset", ["imdb", "dblp", "acm", "freebase"])
    def test_all_hgb_datasets_prepare(self, dataset: str):
        """DataModule sets up correctly for each HGB name."""
        from exp.datasets.hgb import HetSheafDataModule

        data = _make_het_data()
        info = _make_het_info()

        with patch("exp.datasets.hgb._load_hgb_dataset", return_value=(data, info)):
            dm = HetSheafDataModule(dataset, root="/tmp/fake")
            dm.setup()

        assert dm.num_nodes == data.num_nodes, (
            f"Expected {data.num_nodes} nodes, got {dm.num_nodes}"
        )
        assert dm.num_node_types == info.num_node_types, (
            f"Expected {info.num_node_types} node types, got {dm.num_node_types}"
        )


# ---------------------------------------------------------------------------
# GC smoke tests
# ---------------------------------------------------------------------------


def _make_gc_graph(n: int = 8, f: int = 4, num_classes: int = 2) -> Data:
    return Data(
        x=torch.ones(n, f),
        edge_index=torch.randint(0, n, (2, n * 2)),
        y=torch.randint(0, num_classes, (1,)),
        node_type=torch.zeros(n, dtype=torch.long),
        edge_type=torch.zeros(n * 2, dtype=torch.long),
        batch=torch.zeros(n, dtype=torch.long),
    )


def _make_mock_tu(n: int = 20, f: int = 4, num_classes: int = 2, num_nt: int = 1):
    graphs = [_make_gc_graph(f=f, num_classes=num_classes) for _ in range(n)]
    if num_nt > 1:
        for g in graphs:
            assert g.num_nodes is not None, (
                "Graph should have num_nodes defined in _make_mock_tu"
            )
            g.node_type = torch.randint(0, num_nt, (int(g.num_nodes),))

    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=n)
    mock_ds.num_classes = num_classes
    mock_ds.num_node_features = f

    def _getitem(idx):
        if isinstance(idx, torch.Tensor):
            items = [graphs[i] for i in idx.tolist()]
            sub = MagicMock()
            sub.__len__ = MagicMock(return_value=len(items))
            sub.__getitem__ = MagicMock(side_effect=lambda i: items[i])
            sub.__iter__ = MagicMock(return_value=iter(items))
            return sub
        return graphs[idx]

    mock_ds.__getitem__ = MagicMock(side_effect=_getitem)
    mock_ds.__iter__ = MagicMock(return_value=iter(graphs))
    return mock_ds


class TestGCSmoke:
    def test_homo_gc_one_epoch(self):
        from lightning import Trainer

        from exp.datasets.graph_classification import GCSheafDataModule
        from exp.datasets.info import GCDatasetInfo
        from exp.gc_module import GCLightningModule

        cfg, _, _ = _cfg(num_features=4, num_classes=2)
        mock_ds = _make_mock_tu(n=20, f=4, num_classes=2)

        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_ds):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()
            info = cast(GCDatasetInfo, dm.info)
            module = GCLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_acc" in result, (
            f"Expected test_acc in result for homo GC, got {result.keys()}"
        )

    def test_het_gc_one_epoch(self):
        from lightning import Trainer

        from exp.datasets.graph_classification import GCHetSheafDataModule
        from exp.datasets.info import GCDatasetInfo
        from exp.gc_module import GCLightningModule

        cfg, _, _ = _cfg(num_features=1, num_classes=2)
        mock_ds = _make_mock_tu(n=20, f=1, num_classes=2, num_nt=2)

        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_ds):
            dm = GCHetSheafDataModule("mutag", root="/tmp/fake")
            dm.setup()
            info = cast(GCDatasetInfo, dm.info)
            module = GCLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_acc" in result, (
            f"Expected test_acc in result for het GC, got {result.keys()}"
        )

    def test_proteins_het_gc_one_epoch(self):
        """PROTEINS: 1113 graphs, 3 node types - tests heterogeneous GC."""
        from lightning import Trainer

        from exp.datasets.graph_classification import GCHetSheafDataModule
        from exp.datasets.info import GCDatasetInfo
        from exp.gc_module import GCLightningModule

        cfg, _, _ = _cfg(num_features=1, num_classes=2)
        # Mock 20 graphs, each with node_type (3 types)
        mock_ds = _make_mock_tu(n=20, f=1, num_classes=2, num_nt=3)

        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_ds):
            dm = GCHetSheafDataModule("proteins", root="/tmp/fake")
            dm.setup()
            info = cast(GCDatasetInfo, dm.info)
            module = GCLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_acc" in result

    @pytest.mark.parametrize("pooling_name", ["mean", "add", "max"])
    def test_gc_pooling_strategies(self, pooling_name: str):
        """Mean / add / max pooling all train without error."""
        from lightning import Trainer

        from exp.config import GCPooling
        from exp.datasets.graph_classification import GCSheafDataModule
        from exp.datasets.info import GCDatasetInfo
        from exp.gc_module import GCLightningModule

        cfg, _, _ = _cfg(num_features=4, num_classes=2)
        cfg.model.pooling = GCPooling(pooling_name)
        mock_ds = _make_mock_tu(n=20, f=4, num_classes=2)

        with patch("exp.datasets.graph_classification.TUDataset", return_value=mock_ds):
            dm = GCSheafDataModule("collab", root="/tmp/fake")
            dm.setup()
            info = cast(GCDatasetInfo, dm.info)
            module = GCLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)


# ---------------------------------------------------------------------------
# LP smoke test (DataModule pipeline only  no LP LightningModule yet)
# ---------------------------------------------------------------------------


def _make_lp_hetero(
    edge_type: tuple[str, str, str],
    n_src: int = 30,
    n_dst: int = 40,
    n_edges: int = 80,
) -> HeteroData:
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


class TestLPSmoke:
    def test_lastfm_one_epoch(self):
        from lightning import Trainer

        from exp.datasets.link_pred import LastFMDataModule
        from exp.lp_module import LPLightningModule

        hetero = _make_lp_hetero(("user", "to", "artist"))
        cfg, _, _ = _cfg()
        cfg.model.jknet = True
        cfg.model.normalize_output = False

        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=hetero,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake", homogeneous=True)
            dm.prepare_data()
            from exp.datasets.info import LPDatasetInfo

            info = cast(LPDatasetInfo, dm.info)
            module = LPLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_auroc" in result

    def test_movielens_one_epoch(self):
        from lightning import Trainer

        from exp.datasets.link_pred import MovieLensDataModule
        from exp.lp_module import LPLightningModule

        hetero = _make_lp_hetero(("user", "rates", "movie"))
        cfg, _, _ = _cfg()
        cfg.model.jknet = True
        cfg.model.normalize_output = False

        with patch(
            "exp.datasets.link_pred.MovieLensDataModule.download_data",
            return_value=hetero,
        ):
            dm = MovieLensDataModule(data_dir="/tmp/fake", homogeneous=True)
            dm.prepare_data()
            from exp.datasets.info import LPDatasetInfo

            info = cast(LPDatasetInfo, dm.info)
            module = LPLightningModule(cfg, info)
            trainer = Trainer(
                fast_dev_run=True,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(module, dm)
            [result] = trainer.test(module, dm, verbose=False)

        assert "test_auroc" in result

    def test_lastfm_prepare_and_dataloaders(self):
        from exp.datasets.link_pred import LastFMDataModule

        hetero = _make_lp_hetero(("user", "to", "artist"))

        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=hetero,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        # All three splits set
        assert dm.train_data is not None, "Training data not loaded for LastFM"
        assert dm.val_data is not None, "Validation data not loaded for LastFM"
        assert dm.test_data is not None, "Test data not loaded for LastFM"
        # Info populated
        assert dm.info is not None, "DatasetInfo not populated for LastFM"
        assert dm.info.metric == "roc_auc", (
            f"Expected roc_auc for LastFM, got {dm.info.metric}"
        )
        # Dataloaders yield batches
        batch = next(iter(dm.train_dataloader()))
        assert batch is not None, "Train dataloader returned None batch for LastFM"

    def test_movielens_prepare_and_dataloaders(self):
        from exp.datasets.link_pred import MovieLensDataModule

        hetero = _make_lp_hetero(("user", "rates", "movie"))

        with patch(
            "exp.datasets.link_pred.MovieLensDataModule.download_data",
            return_value=hetero,
        ):
            dm = MovieLensDataModule(data_dir="/tmp/fake")
            dm.prepare_data()

        assert dm.train_data is not None, "Training data not loaded for MovieLens"
        assert dm.info is not None, "DatasetInfo not populated for MovieLens"
        assert dm.info.target == (
            "user",
            "rates",
            "movie",
        ), f"Incorrect target for MovieLens: {dm.info.target}"
        batch = next(iter(dm.train_dataloader()))
        assert batch is not None, "Train dataloader returned None batch for MovieLens"

    def test_lastfm_homogeneous_mode(self):
        from exp.datasets.link_pred import LastFMDataModule

        hetero = _make_lp_hetero(("user", "to", "artist"))

        with patch(
            "exp.datasets.link_pred.LastFMDataModule.download_data",
            return_value=hetero,
        ):
            dm = LastFMDataModule(data_dir="/tmp/fake", homogeneous=True)
            dm.prepare_data()

        assert dm.train_data is not None, (
            "Training data not loaded for homogeneous LastFM"
        )
        assert dm.info is not None, "DatasetInfo not populated for homogeneous LastFM"
        # In homogeneous mode the info should report type counts from to_homogeneous()
        assert dm.info.num_node_types >= 1, (
            "Expected at least 1 node type in homogeneous mode"
        )
