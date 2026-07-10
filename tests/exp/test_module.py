# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/module.py -- SheafLightningModule training and evaluation logic."""

from __future__ import annotations

from typing import Literal as _Lit
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from exp.config import Config, ModelConfig, OptimConfig, RegConfig
from exp.data import DatasetInfo
from exp.module import SheafLightningModule
from sheaf_mpnn.nsd import NSDModel

_VariantStr = _Lit[
    "diagonal",
    "general",
    "orthogonal",
    "low_rank",
    "general_attention",
    "orthogonal_attention",
]

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(
    variant: _VariantStr = "general",
    stalk_dim: int = 2,
    hidden_dim: int = 4,
    num_layers: int = 1,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    input_dropout: float = 0.0,
    dropout: float = 0.0,
) -> Config:
    return Config(
        model=ModelConfig(
            variant=variant,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            alpha=1.0,
        ),
        reg=RegConfig(input_dropout=input_dropout, dropout=dropout),
        optim=OptimConfig(lr=lr, weight_decay=weight_decay),
    )


def _make_info(
    metric: str = "acc", num_features: int = 8, num_classes: int = 3
) -> DatasetInfo:
    return DatasetInfo(
        name="synthetic",
        num_features=num_features,
        num_classes=num_classes,
        num_splits=10,
        metric=metric,
        split_type="npz_file",
    )


def _make_batch(
    num_nodes: int = 20, num_features: int = 8, num_classes: int = 3
) -> Data:
    torch.manual_seed(7)
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    y = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:12] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[12:16] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[16:] = True
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _make_module(metric: str = "acc", **cfg_kwargs) -> SheafLightningModule:
    cfg = _make_config(**cfg_kwargs)
    info = _make_info(metric=metric)
    mod = SheafLightningModule(cfg, info)
    mod.log = MagicMock()  # ty: ignore[invalid-assignment]  test mocking
    return mod


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestModuleConstruction:
    @pytest.mark.parametrize("variant", ["diagonal", "general", "orthogonal"])
    def test_all_variants_build_nsd_model(self, variant):
        mod = _make_module(variant=variant)
        assert isinstance(mod.model, NSDModel)

    def test_model_config_matches_cfg(self):
        mod = _make_module(stalk_dim=3, hidden_dim=8, num_layers=2)
        assert mod.cfg.model.stalk_dim == 3
        assert mod.cfg.model.hidden_dim == 8
        assert mod.cfg.model.num_layers == 2

    def test_encoder_has_correct_in_out(self):
        mod = _make_module(stalk_dim=2, hidden_dim=4)
        # encoder: in_features -> d * hidden_dim
        assert mod.model.encoder.in_features == 8  # num_features
        assert mod.model.encoder.out_features == 8  # 2 * 4

    def test_decoder_has_correct_out(self):
        mod = _make_module(stalk_dim=2, hidden_dim=4)
        # decoder: d*hidden_dim -> num_classes
        assert mod.model.decoder.out_features == 3  # num_classes

    def test_unknown_model_type_raises_value_error(self):
        cfg = _make_config()
        cfg.model.type = "totally_unknown"  # type: ignore
        info = _make_info()
        with pytest.raises(ValueError, match="Unknown model type"):
            SheafLightningModule(cfg, info)


# ---------------------------------------------------------------------------
# training_step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    @pytest.fixture
    def module(self):
        return _make_module()

    @pytest.fixture
    def batch(self):
        return _make_batch()

    def test_returns_scalar_tensor(self, module, batch):
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_loss_is_finite(self, module, batch):
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_loss_is_positive(self, module, batch):
        loss = module.training_step(batch, 0)
        assert loss.item() > 0

    def test_loss_logs_train_loss(self, module, batch):
        module.training_step(batch, 0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "train_loss" in logged_keys

    def test_backward_produces_gradients(self, module, batch):
        loss = module.training_step(batch, 0)
        loss.backward()
        assert module.model.encoder.weight.grad is not None
        assert module.model.encoder.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("variant", ["diagonal", "general", "orthogonal"])
    def test_all_variants_produce_finite_loss(self, variant, batch):
        mod = _make_module(variant=variant)
        loss = mod.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_input_dropout_applied_in_training(self, batch):
        mod = _make_module(input_dropout=0.9, dropout=0.9)
        loss = mod.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_train_mode_is_stochastic_with_dropout(self, batch):
        """High dropout in train mode must produce different logits each call."""
        mod = _make_module(input_dropout=0.9, dropout=0.9)
        mod.train()
        with torch.no_grad():
            out1 = mod.model(batch.x, batch.edge_index)
            out2 = mod.model(batch.x, batch.edge_index)
        assert not torch.allclose(out1, out2)

    def test_eval_mode_suppresses_dropout(self, batch):
        """eval() must make model output deterministic regardless of dropout rate."""
        mod = _make_module(input_dropout=0.9, dropout=0.9)
        mod.eval()
        with torch.no_grad():
            out1 = mod.model(batch.x, batch.edge_index)
            out2 = mod.model(batch.x, batch.edge_index)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# validation_step / test_step
# ---------------------------------------------------------------------------


class TestEvalSteps:
    @pytest.fixture
    def module(self):
        return _make_module(metric="acc")

    @pytest.fixture
    def batch(self):
        return _make_batch()

    def test_validation_step_logs_val_loss(self, module, batch):
        module.validation_step(batch, 0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "val_loss" in logged_keys

    def test_validation_step_logs_val_acc(self, module, batch):
        module.validation_step(batch, 0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "val_acc" in logged_keys

    def test_test_step_logs_test_loss(self, module, batch):
        module.test_step(batch, 0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "test_loss" in logged_keys

    def test_test_step_logs_test_acc(self, module, batch):
        module.test_step(batch, 0)
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "test_acc" in logged_keys

    def test_roc_auc_metric_logged_correctly(self, batch):
        mod = _make_module(metric="roc_auc")
        # Guarantee both classes present so ROC-AUC is well-defined.
        n = batch.num_nodes
        batch.y = torch.cat(
            [
                torch.zeros(n // 2, dtype=torch.long),
                torch.ones(n - n // 2, dtype=torch.long),
            ]
        )
        mod.validation_step(batch, 0)
        logged_keys = [call.args[0] for call in cast(MagicMock, mod.log).call_args_list]
        assert "val_roc_auc" in logged_keys

    def test_logged_val_loss_is_finite(self, module, batch):
        module.validation_step(batch, 0)
        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        val_loss = logged["val_loss"]
        assert torch.isfinite(val_loss.detach())

    def test_eval_step_is_deterministic(self, batch):
        """validation_step in eval mode must produce the same logged metrics twice."""
        mod = _make_module(metric="acc", input_dropout=0.9, dropout=0.9)
        mod.eval()
        mock_log = cast(MagicMock, mod.log)
        with torch.no_grad():
            mod.validation_step(batch, 0)
            logged1 = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            mock_log.reset_mock()
            mod.validation_step(batch, 0)
            logged2 = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
        assert float(logged1["val_loss"]) == pytest.approx(float(logged2["val_loss"]))


# ---------------------------------------------------------------------------
# NaN / inf logit handling in _eval_step
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def _nan_forward(self, num_nodes: int, num_classes: int):
        return torch.full((num_nodes, num_classes), float("nan"))

    def test_nan_logits_logs_inf_val_loss_for_acc(self):
        mod = _make_module(metric="acc")
        batch = _make_batch()
        assert batch.num_nodes is not None
        nan_out = self._nan_forward(batch.num_nodes, 3)
        with patch.object(mod.model, "forward", return_value=nan_out):
            mod._eval_step(batch, "val_mask", "val")
        logged = {c.args[0]: c.args[1] for c in cast(MagicMock, mod.log).call_args_list}
        assert logged["val_loss"].item() == float("inf")

    def test_nan_logits_logs_zero_acc(self):
        mod = _make_module(metric="acc")
        batch = _make_batch()
        assert batch.num_nodes is not None
        nan_out = self._nan_forward(batch.num_nodes, 3)
        with patch.object(mod.model, "forward", return_value=nan_out):
            mod._eval_step(batch, "val_mask", "val")
        logged = {c.args[0]: c.args[1] for c in cast(MagicMock, mod.log).call_args_list}
        assert logged["val_acc"].item() == pytest.approx(0.0)

    def test_nan_logits_logs_half_roc_auc(self):
        mod = _make_module(metric="roc_auc")
        batch = _make_batch()
        assert batch.num_nodes is not None
        nan_out = self._nan_forward(batch.num_nodes, 3)
        with patch.object(mod.model, "forward", return_value=nan_out):
            mod._eval_step(batch, "val_mask", "val")
        logged = {c.args[0]: c.args[1] for c in cast(MagicMock, mod.log).call_args_list}
        assert logged["val_roc_auc"].item() == pytest.approx(0.5)

    def test_nan_logits_early_returns_without_mask_access(self):
        mod = _make_module(metric="acc")
        batch = _make_batch()
        assert batch.num_nodes is not None
        nan_out = self._nan_forward(batch.num_nodes, 3)
        with patch.object(mod.model, "forward", return_value=nan_out):
            # Should not raise; only the two early-return log calls should be made
            mod._eval_step(batch, "val_mask", "val")
        assert cast(MagicMock, mod.log).call_count == 2


# ---------------------------------------------------------------------------
# _compute_metric -- accuracy
# ---------------------------------------------------------------------------


class TestComputeMetricAcc:
    @pytest.fixture
    def module(self):
        return _make_module(metric="acc")

    def test_perfect_accuracy(self, module):
        # logits[i] argmax == labels[i] for all nodes
        logits = torch.eye(3).repeat(3, 1)  # shape (9, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        mask = torch.ones(9, dtype=torch.bool)
        assert module._compute_metric(logits, labels, mask) == pytest.approx(1.0)

    def test_zero_accuracy(self, module):
        # All logits push to class 0, all labels are class 1
        logits = torch.zeros(5, 3)
        logits[:, 0] = 10.0
        labels = torch.ones(5, dtype=torch.long)
        mask = torch.ones(5, dtype=torch.bool)
        assert module._compute_metric(logits, labels, mask) == pytest.approx(0.0)

    def test_partial_mask_uses_only_masked_nodes(self, module):
        logits = torch.eye(3)  # 3 nodes: correct predictions
        labels = torch.tensor([0, 1, 2])
        mask = torch.tensor([True, True, False])  # ignore node 2
        assert module._compute_metric(logits, labels, mask) == pytest.approx(1.0)

    def test_metric_is_float(self, module):
        logits = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))
        mask = torch.ones(10, dtype=torch.bool)
        result = module._compute_metric(logits, labels, mask)
        assert isinstance(result, float)

    def test_metric_in_unit_interval(self, module):
        torch.manual_seed(0)
        logits = torch.randn(20, 3)
        labels = torch.randint(0, 3, (20,))
        mask = torch.ones(20, dtype=torch.bool)
        result = module._compute_metric(logits, labels, mask)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _compute_metric -- ROC-AUC
# ---------------------------------------------------------------------------


class TestComputeMetricROCAUC:
    @pytest.fixture
    def module(self):
        return _make_module(metric="roc_auc")

    def test_returns_float_in_unit_interval(self, module):
        torch.manual_seed(1)
        logits = torch.randn(20, 2)
        labels = torch.randint(0, 2, (20,))
        mask = torch.ones(20, dtype=torch.bool)
        result = module._compute_metric(logits, labels, mask)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_single_class_in_mask_returns_half(self, module):
        logits = torch.randn(10, 2)
        labels = torch.zeros(10, dtype=torch.long)  # only class 0 -> np.unique < 2
        mask = torch.ones(10, dtype=torch.bool)
        result = module._compute_metric(logits, labels, mask)
        assert result == pytest.approx(0.5)

    def test_multiclass_roc_auc(self, module):
        # Create an info with 3 classes and roc_auc
        cfg = _make_config()
        info = DatasetInfo("syn", 8, 3, 10, "roc_auc", "npz_file")
        mod = SheafLightningModule(cfg, info)
        mod.log = MagicMock()  # ty: ignore[invalid-assignment]  test mocking
        torch.manual_seed(2)
        logits = torch.randn(30, 3)
        labels = torch.randint(0, 3, (30,))
        mask = torch.ones(30, dtype=torch.bool)
        result = mod._compute_metric(logits, labels, mask)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    def test_returns_adam_optimizer(self):
        mod = _make_module(lr=0.01, weight_decay=5e-4)
        optimizer = mod.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)

    def test_lr_matches_config(self):
        mod = _make_module(lr=1e-3)
        optimizer = mod.configure_optimizers()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_weight_decay_matches_config(self):
        mod = _make_module(weight_decay=1e-5)
        optimizer = mod.configure_optimizers()
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)

    def test_optimizer_covers_all_parameters(self):
        mod = _make_module()
        optimizer = mod.configure_optimizers()
        opt_param_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        model_param_ids = {id(p) for p in mod.parameters()}
        assert model_param_ids.issubset(opt_param_ids)
