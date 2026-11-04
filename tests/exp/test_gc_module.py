# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/gc_module.py -- GCLightningModule training and pooling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

from exp.config import Config, GCPooling, ModelConfig, ModelType, OptimConfig, RegConfig
from exp.datasets.info import GCDatasetInfo
from exp.gc_module import GCLightningModule, HetGCLightningModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gc_config(
    variant: str = "diagonal",
    stalk_dim: int = 2,
    hidden_dim: int = 4,
    num_layers: int = 1,
    pooling: GCPooling = GCPooling.MEAN,
) -> Config:
    return Config(
        model=ModelConfig(
            variant=variant,  # ty: ignore[invalid-argument-type]
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=1.0,
            pooling=pooling,
        ),
        reg=RegConfig(input_dropout=0.0, dropout=0.0),
        optim=OptimConfig(lr=0.01, weight_decay=5e-4),
    )


def _make_gc_info(
    num_features: int = 4,
    num_classes: int = 2,
) -> GCDatasetInfo:
    return GCDatasetInfo(
        name="mutag",
        num_features=num_features,
        num_classes=num_classes,
        num_splits=10,
        metric="acc",
        split_type="seed",
        num_node_types=7,
        num_edge_types=4,
        task="binary",
    )


def _make_gc_batch(
    num_nodes_per_graph: int = 4,
    num_graphs: int = 2,
    num_features: int = 4,
    num_classes: int = 2,
) -> Data:
    """Create a batched graph for GC testing."""
    torch.manual_seed(42)
    total_nodes = num_nodes_per_graph * num_graphs
    x = torch.randn(total_nodes, num_features)

    # Each graph is a simple chain
    edges_per_graph = num_nodes_per_graph - 1
    src_list = []
    dst_list = []
    for g in range(num_graphs):
        offset = g * num_nodes_per_graph
        for i in range(edges_per_graph):
            src_list.append(offset + i)
            dst_list.append(offset + i + 1)
            src_list.append(offset + i + 1)
            dst_list.append(offset + i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    batch = torch.repeat_interleave(
        torch.arange(num_graphs), torch.tensor([num_nodes_per_graph] * num_graphs)
    )
    y = torch.randint(0, num_classes, (num_graphs,))

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        batch=batch,
        num_nodes=total_nodes,
    )
    return data


# ---------------------------------------------------------------------------
# SAGPooling construction
# ---------------------------------------------------------------------------


class TestGCLightningModuleSAGPooling:
    def test_sag_pooling_attribute_created(self):
        """Line 85: sag_pool is created when pooling=GCPooling.SAG."""
        cfg = _make_gc_config(pooling=GCPooling.SAG)
        info = _make_gc_info()
        module = GCLightningModule(cfg, info)
        assert hasattr(module, "sag_pool"), (
            "sag_pool attribute should be set for SAG pooling"
        )

    def test_non_sag_pooling_no_sag_attribute(self):
        """MEAN pooling does not create sag_pool."""
        cfg = _make_gc_config(pooling=GCPooling.MEAN)
        info = _make_gc_info()
        module = GCLightningModule(cfg, info)
        assert not hasattr(module, "sag_pool"), (
            "sag_pool should not exist for MEAN pooling"
        )

    def test_sag_pool_has_correct_in_channels(self):
        """sag_pool in_channels equals stalk_dim * hidden_dim (raw stalk features)."""
        cfg = _make_gc_config(stalk_dim=2, hidden_dim=4, pooling=GCPooling.SAG)
        info = _make_gc_info(num_classes=3)
        module = GCLightningModule(cfg, info)
        assert module.sag_pool.in_channels == 2 * 4  # context_dim


# ---------------------------------------------------------------------------
# SAGPooling _pool()
# ---------------------------------------------------------------------------


class TestGCLightningModulePoolSAG:
    def test_sag_pool_output_shape(self):
        """Lines 105-108: _pool with SAG returns [num_graphs, num_classes]."""
        cfg = _make_gc_config(stalk_dim=2, hidden_dim=4, pooling=GCPooling.SAG)
        info = _make_gc_info(num_features=4, num_classes=2)
        module = GCLightningModule(cfg, info)
        module.eval()
        module.log = MagicMock()  # ty: ignore[invalid-assignment]

        batch = _make_gc_batch(
            num_nodes_per_graph=4, num_graphs=2, num_features=4, num_classes=2
        )
        assert batch.num_nodes is not None
        assert batch.edge_index is not None
        assert batch.batch is not None
        context_dim = 2 * 4  # stalk_dim * hidden_dim
        node_features = torch.randn(int(batch.num_nodes), context_dim)

        with torch.no_grad():
            result = module._pool(node_features, batch.edge_index, batch.batch)

        assert result.shape == (
            2,
            context_dim,
        ), f"Expected (2, {context_dim}), got {result.shape}"

    def test_sag_pool_training_step_runs(self):
        """Full training_step with SAG pooling should not raise."""
        cfg = _make_gc_config(stalk_dim=2, hidden_dim=4, pooling=GCPooling.SAG)
        info = _make_gc_info(num_features=4, num_classes=2)
        module = GCLightningModule(cfg, info)
        module.log = MagicMock()  # ty: ignore[invalid-assignment]

        batch = _make_gc_batch(
            num_nodes_per_graph=4, num_graphs=2, num_features=4, num_classes=2
        )
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Other pooling strategies
# ---------------------------------------------------------------------------


class TestGCPoolingStrategies:
    @pytest.mark.parametrize("pooling", [GCPooling.MEAN, GCPooling.ADD, GCPooling.MAX])
    def test_pool_output_shape(self, pooling):
        """All non-SAG pooling strategies produce [num_graphs, num_classes]."""
        cfg = _make_gc_config(pooling=pooling)
        info = _make_gc_info(num_features=4, num_classes=2)
        module = GCLightningModule(cfg, info)
        module.eval()

        batch = _make_gc_batch(
            num_nodes_per_graph=4, num_graphs=2, num_features=4, num_classes=2
        )
        assert batch.num_nodes is not None
        context_dim = 2 * 4  # stalk_dim * hidden_dim from _make_gc_config defaults
        node_features = torch.randn(int(batch.num_nodes), context_dim)

        with torch.no_grad():
            assert batch.edge_index is not None
            assert batch.batch is not None
            result = module._pool(node_features, batch.edge_index, batch.batch)

        assert result.shape == (
            2,
            context_dim,
        ), f"Expected (2, {context_dim}) for pooling={pooling}, got {result.shape}"


# ---------------------------------------------------------------------------
# GCLightningModule with SHEAF_POOL
# ---------------------------------------------------------------------------


class TestGCLightningModuleSheafPool:
    def _make_module(
        self,
        stalk_dim: int = 3,
        hidden_dim: int = 6,
        num_classes: int = 3,
        model_type: str = "nsd",
        variant: str = "diagonal",
    ) -> GCLightningModule:
        cfg = Config(
            model=ModelConfig(
                type=ModelType(model_type),
                variant=variant,  # ty: ignore[invalid-argument-type]
                stalk_dim=stalk_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                alpha=1.0,
                pooling=GCPooling.SHEAF_POOL,
            ),
            reg=RegConfig(input_dropout=0.0, dropout=0.0),
            optim=OptimConfig(lr=0.01, weight_decay=5e-4),
        )
        info = _make_gc_info(num_features=4, num_classes=num_classes)
        return GCLightningModule(cfg, info)

    def test_sheaf_pool_attributes_created(self):
        module = self._make_module()
        assert hasattr(module, "sheaf_pool")
        assert hasattr(module, "gc_classifier")

    def test_sheaf_pool_dims(self):
        module = self._make_module(stalk_dim=3, hidden_dim=6, num_classes=4)
        from sheaf_mpnn.sheaf_pool import SheafPool

        assert isinstance(module.sheaf_pool, SheafPool)
        assert module.sheaf_pool.stalk_dim == 3
        assert module.sheaf_pool.channel_dim == 6
        assert module.gc_classifier.out_features == 4

    @pytest.mark.parametrize(
        "model_type,variant",
        [("nsd", "diagonal"), ("nsp", "diagonal"), ("sheaf_attn", "sheafan")],
    )
    def test_training_step_runs(self, model_type, variant):
        module = self._make_module(model_type=model_type, variant=variant)
        module.log = MagicMock()  # ty: ignore[invalid-assignment]
        batch = _make_gc_batch(num_nodes_per_graph=5, num_graphs=2, num_features=4)
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize(
        "model_type,variant",
        [("nsd", "diagonal"), ("nsp", "diagonal"), ("sheaf_attn", "sheafan")],
    )
    def test_graph_logits_shape(self, model_type, variant):
        module = self._make_module(
            model_type=model_type, num_classes=3, variant=variant
        )
        module.eval()
        batch = _make_gc_batch(num_nodes_per_graph=5, num_graphs=2, num_features=4)
        assert batch.x is not None
        assert batch.edge_index is not None
        assert batch.batch is not None
        with torch.no_grad():
            logits = module._graph_logits(batch.x, batch.edge_index, batch.batch)
        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()

    def test_gradient_flows(self):
        module = self._make_module()
        module.log = MagicMock()  # ty: ignore[invalid-assignment]
        batch = _make_gc_batch(num_nodes_per_graph=4, num_graphs=2, num_features=4)
        loss = module.training_step(batch, 0)
        loss.backward()
        # Anchor and token in SheafPool should receive gradients
        assert module.sheaf_pool.anchor.grad is not None
        assert module.sheaf_pool.token.grad is not None
        assert module.gc_classifier.weight.grad is not None

    def test_no_sag_pool_created(self):
        module = self._make_module()
        assert not hasattr(module, "sag_pool")


# ---------------------------------------------------------------------------
# HetGCLightningModule with SHEAF_POOL
# ---------------------------------------------------------------------------


class TestHetGCLightningModuleSheafPool:
    def _make_het_module(
        self, stalk_dim: int = 3, hidden_dim: int = 6
    ) -> HetGCLightningModule:
        cfg = Config(
            model=ModelConfig(
                type=ModelType.HET_NSD,
                variant="diagonal",
                stalk_dim=stalk_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                alpha=1.0,
                pooling=GCPooling.SHEAF_POOL,
                num_node_types=3,
                num_edge_types=2,
                use_features=True,
            ),
            reg=RegConfig(input_dropout=0.0, dropout=0.0),
            optim=OptimConfig(lr=0.01, weight_decay=5e-4),
        )
        info = _make_gc_info(num_features=4, num_classes=2)
        return HetGCLightningModule(cfg, info)

    def test_het_training_step_runs(self):
        module = self._make_het_module()
        module.log = MagicMock()  # ty: ignore[invalid-assignment]
        batch = _make_gc_batch(num_nodes_per_graph=5, num_graphs=2, num_features=4)
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    def test_het_graph_logits_shape(self):
        module = self._make_het_module()
        module.eval()
        batch = _make_gc_batch(num_nodes_per_graph=5, num_graphs=2, num_features=4)
        assert batch.x is not None
        assert batch.edge_index is not None
        assert batch.batch is not None
        with torch.no_grad():
            logits = module._graph_logits(batch.x, batch.edge_index, batch.batch)
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()
