# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/lp_module.py -- LPLightningModule training and evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from exp.config import Config, ModelConfig, OptimConfig, RegConfig
from exp.datasets.info import LPDatasetInfo
from exp.lp_module import HetLPLightningModule, LPLightningModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lp_config(
    stalk_dim: int = 2,
    hidden_dim: int = 4,
    num_layers: int = 1,
) -> Config:
    return Config(
        model=ModelConfig(
            variant="diagonal",
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=1.0,
        ),
        reg=RegConfig(input_dropout=0.0, dropout=0.0),
        optim=OptimConfig(lr=0.01, weight_decay=5e-4),
    )


def _make_lp_info(num_features: int = 4) -> LPDatasetInfo:
    return LPDatasetInfo(
        name="lastfm",
        num_features=num_features,
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


def _make_lp_batch(
    num_nodes: int = 10,
    num_features: int = 4,
    num_sup_edges: int = 8,
    all_same_label: bool = False,
) -> Data:
    """Create a link-prediction batch."""
    torch.manual_seed(0)
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_label_index = torch.randint(0, num_nodes, (2, num_sup_edges))
    if all_same_label:
        edge_label = torch.ones(num_sup_edges, dtype=torch.float)
    else:
        edge_label = torch.randint(0, 2, (num_sup_edges,)).float()
        # Ensure both classes present
        edge_label[0] = 0.0
        edge_label[1] = 1.0
    return Data(
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        num_nodes=num_nodes,
    )


def _make_het_lp_batch(
    num_nodes: int = 10,
    num_features: int = 4,
    num_sup_edges: int = 8,
    all_same_label: bool = False,
    with_types: bool = True,
) -> Data:
    """Create a link-prediction batch with optional node/edge type tensors."""
    data = _make_lp_batch(num_nodes, num_features, num_sup_edges, all_same_label)
    if with_types:
        data.node_type = torch.randint(0, 2, (num_nodes,))
        # _make_lp_batch creates num_nodes * 2 message-passing edges.
        data.edge_type = torch.zeros(num_nodes * 2, dtype=torch.long)
    return data


class _RecordingEncoder(nn.Module):
    """Stub encoder recording the (node_types, edge_types) it was called with."""

    def __init__(self, num_nodes: int, emb_dim: int) -> None:
        super().__init__()
        self.z = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.calls: list[tuple] = []

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        self.calls.append((node_types, edge_types))
        return self.z


def _make_het_module(num_nodes: int = 10) -> tuple[HetLPLightningModule, MagicMock]:
    """Build a HetLPLightningModule with a stub encoder and a mocked logger."""
    cfg = _make_lp_config()
    info = _make_lp_info()
    module = HetLPLightningModule(cfg, info)
    emb_dim = cfg.model.hidden_dim * cfg.model.stalk_dim
    module.model = _RecordingEncoder(num_nodes, emb_dim)
    mock_log = MagicMock()
    module.log = mock_log  # ty: ignore[invalid-assignment]
    return module, mock_log


# ---------------------------------------------------------------------------
# TestLPLightningModule construction
# ---------------------------------------------------------------------------


class TestLPLightningModuleConstruction:
    def test_builds_model(self):
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        assert module.model is not None

    def test_stores_cfg_and_info(self):
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        assert module.cfg is cfg
        assert module.info is info


# ---------------------------------------------------------------------------
# TestLPLightningModule training_step
# ---------------------------------------------------------------------------


class TestLPTrainingStep:
    def test_returns_scalar_tensor(self):
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        module.log = MagicMock()  # ty: ignore[invalid-assignment]
        batch = _make_lp_batch()
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_logs_train_loss(self):
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        mock_log = MagicMock()
        module.log = mock_log  # ty: ignore[invalid-assignment]
        batch = _make_lp_batch()
        module.training_step(batch, 0)
        logged_keys = [call.args[0] for call in mock_log.call_args_list]
        assert "train_loss" in logged_keys


# ---------------------------------------------------------------------------
# TestLPLightningModule eval_step -- degenerate labels (line 131)
# ---------------------------------------------------------------------------


class TestLPEvalStepDegenerateLabels:
    def test_all_same_labels_returns_auroc_half_aupr_zero(self):
        """Line 131: when np.unique(labels).size < 2, auroc=0.5 and aupr=0.0."""
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        mock_log = MagicMock()
        module.log = mock_log  # ty: ignore[invalid-assignment]

        # All labels are 1 (same class) -> np.unique has size 1
        batch = _make_lp_batch(all_same_label=True)
        module.validation_step(batch, 0)

        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        assert "val_auroc" in logged, "val_auroc should be logged"
        assert "val_aupr" in logged, "val_aupr should be logged"
        assert logged["val_auroc"] == pytest.approx(0.5), (
            f"Expected auroc=0.5 for degenerate labels, got {logged['val_auroc']}"
        )
        assert logged["val_aupr"] == pytest.approx(0.0), (
            f"Expected aupr=0.0 for degenerate labels, got {logged['val_aupr']}"
        )

    def test_normal_labels_compute_auroc(self):
        """When both classes present, auroc is computed normally."""
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        mock_log = MagicMock()
        module.log = mock_log  # ty: ignore[invalid-assignment]

        batch = _make_lp_batch(all_same_label=False)
        module.validation_step(batch, 0)

        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        assert "val_auroc" in logged
        # auroc should be in [0, 1]
        assert 0.0 <= logged["val_auroc"] <= 1.0


# ---------------------------------------------------------------------------
# TestLPLightningModule test_step degenerate labels
# ---------------------------------------------------------------------------


class TestLPTestStepDegenerateLabels:
    def test_all_same_labels_in_test_step(self):
        """test_step also hits the degenerate branch."""
        cfg = _make_lp_config()
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        mock_log = MagicMock()
        module.log = mock_log  # ty: ignore[invalid-assignment]

        batch = _make_lp_batch(all_same_label=True)
        module.test_step(batch, 0)

        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        assert "test_auroc" in logged
        assert logged["test_auroc"] == pytest.approx(0.5)
        assert logged["test_aupr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestLPLightningModule configure_optimizers
# ---------------------------------------------------------------------------


class TestLPConfigureOptimizers:
    def test_returns_adam(self):
        cfg = _make_lp_config(stalk_dim=2, hidden_dim=4)
        info = _make_lp_info()
        module = LPLightningModule(cfg, info)
        opt = module.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)


# ---------------------------------------------------------------------------
# TestHetLPLightningModule -- heterogeneous training and evaluation
# ---------------------------------------------------------------------------


class TestHetLPTrainingStep:
    def test_returns_scalar_and_forwards_types(self):
        """training_step must pass batch.node_type/edge_type to the model."""
        module, mock_log = _make_het_module()
        batch = _make_het_lp_batch()
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        [(node_types, edge_types)] = module.model.calls
        assert torch.equal(node_types, batch.node_type)
        assert torch.equal(edge_types, batch.edge_type)
        logged_keys = [call.args[0] for call in mock_log.call_args_list]
        assert "train_loss" in logged_keys

    def test_missing_types_default_to_none(self):
        """Batches without node_type/edge_type must pass None to the model."""
        module, _ = _make_het_module()
        batch = _make_het_lp_batch(with_types=False)
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)
        [(node_types, edge_types)] = module.model.calls
        assert node_types is None
        assert edge_types is None


class TestHetLPEvalStep:
    def test_validation_step_logs_metrics(self):
        """val_loss/val_auroc/val_aupr are logged for two-class labels."""
        module, mock_log = _make_het_module()
        batch = _make_het_lp_batch(all_same_label=False)
        module.validation_step(batch, 0)
        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        assert "val_loss" in logged
        assert "val_auroc" in logged
        assert "val_aupr" in logged
        assert 0.0 <= logged["val_auroc"] <= 1.0
        assert 0.0 <= logged["val_aupr"] <= 1.0

    def test_test_step_degenerate_labels(self):
        """Single-class labels fall back to auroc=0.5 and aupr=0.0."""
        module, mock_log = _make_het_module()
        batch = _make_het_lp_batch(all_same_label=True)
        module.test_step(batch, 0)
        logged = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
        assert logged["test_auroc"] == pytest.approx(0.5)
        assert logged["test_aupr"] == pytest.approx(0.0)

    def test_eval_step_forwards_types(self):
        """_eval_step must pass batch.node_type/edge_type to the model."""
        module, _ = _make_het_module()
        batch = _make_het_lp_batch()
        module.validation_step(batch, 0)
        [(node_types, edge_types)] = module.model.calls
        assert torch.equal(node_types, batch.node_type)
        assert torch.equal(edge_types, batch.edge_type)
