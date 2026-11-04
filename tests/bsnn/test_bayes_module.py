# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for BayesSheafLightningModule -- ELBO training, KL annealing, and
ensembled evaluation with uncertainty metrics.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch_geometric.data import Data

from exp.config import Config, ModelConfig, ModelType, OptimConfig, RegConfig
from exp.data import DatasetInfo
from exp.module import (
    BayesSheafLightningModule,
    SheafLightningModule,
    select_module_cls,
)
from sheaf_mpnn.bsnn import BSNNModel

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(**model_kwargs) -> Config:
    defaults: dict = {
        "type": ModelType.BSNN,
        "variant": "diagonal",
        "stalk_dim": 2,
        "hidden_dim": 4,
        "num_layers": 1,
    }
    defaults.update(model_kwargs)
    return Config(
        model=ModelConfig(**defaults),
        reg=RegConfig(),
        optim=OptimConfig(lr=0.01, weight_decay=5e-4),
    )


def _make_info(num_features: int = 8, num_classes: int = 3) -> DatasetInfo:
    return DatasetInfo(
        name="synthetic",
        num_features=num_features,
        num_classes=num_classes,
        num_splits=10,
        metric="acc",
        split_type="npz_file",
    )


def _make_batch(
    num_nodes: int = 20, num_features: int = 8, num_classes: int = 3
) -> Data:
    torch.manual_seed(7)
    x = torch.randn(num_nodes, num_features)
    # Symmetric connectivity: BSNN pairs each edge with its reverse.
    src = torch.randint(0, num_nodes, (num_nodes * 2,))
    dst = torch.randint(0, num_nodes, (num_nodes * 2,))
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
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


def _make_module(**model_kwargs) -> BayesSheafLightningModule:
    torch.manual_seed(42)
    mod = BayesSheafLightningModule(_make_config(**model_kwargs), _make_info())
    mod.log = MagicMock()  # ty: ignore[invalid-assignment]  test mocking
    return mod


# ---------------------------------------------------------------------------
# Construction and dispatch
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("variant", ["diagonal", "general", "orthogonal"])
    def test_builds_bsnn_model(self, variant):
        mod = _make_module(variant=variant)
        assert isinstance(mod.model, BSNNModel)

    def test_select_module_cls(self):
        assert select_module_cls(_make_config()) is BayesSheafLightningModule
        nsd_cfg = Config(model=ModelConfig(type=ModelType.NSD))
        assert select_module_cls(nsd_cfg) is SheafLightningModule

    def test_optimizer_groups_sheaf_learner(self):
        mod = _make_module()
        mod.cfg.optim.sheaf_decay = 1e-6
        opt = mod.configure_optimizers()
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 1e-6
        learner_params = {id(p) for p in mod.model.sheaf_learner.parameters()}
        group_params = {id(p) for p in opt.param_groups[0]["params"]}
        assert learner_params <= group_params


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------


class TestKLBeta:
    def test_cyclical_starts_near_zero(self):
        mod = _make_module(kl_anneal="cyclical", kl_weight=1.0, kl_cycle=40)
        assert mod.kl_beta() < 1e-3  # epoch 0: sigmoid(-10)

    def test_cyclical_reaches_half_weight_mid_cycle(self):
        mod = _make_module(kl_anneal="cyclical", kl_weight=2.0, kl_cycle=40)
        with patch.object(
            BayesSheafLightningModule,
            "current_epoch",
            new_callable=PropertyMock,
            return_value=20,
        ):
            assert mod.kl_beta() == pytest.approx(1.0, rel=1e-5)

    def test_cyclical_restarts_each_cycle(self):
        mod = _make_module(kl_anneal="cyclical", kl_weight=1.0, kl_cycle=40)
        with patch.object(
            BayesSheafLightningModule,
            "current_epoch",
            new_callable=PropertyMock,
            return_value=40,
        ):
            assert mod.kl_beta() < 1e-3

    def test_constant_schedule(self):
        mod = _make_module(kl_anneal="constant", kl_weight=0.5)
        assert mod.kl_beta() == pytest.approx(0.5)

    def test_disabled_kl(self):
        mod = _make_module(use_kl=False, kl_anneal="constant", kl_weight=0.5)
        assert mod.kl_beta() == 0.0


# ---------------------------------------------------------------------------
# Training step (ELBO)
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_returns_finite_loss_and_logs_elbo_terms(self):
        mod = _make_module(kl_anneal="constant", kl_weight=1.0)
        loss = mod.training_step(_make_batch(), 0)
        assert torch.isfinite(loss)
        logged = [c.args[0] for c in cast(MagicMock, mod.log).call_args_list]
        assert {"train_loss", "train_ce", "train_kl", "kl_beta"} <= set(logged)

    def test_kl_term_increases_loss(self):
        # With a constant unit KL weight the total loss must exceed the CE
        # whenever the KL is positive.
        mod = _make_module(kl_anneal="constant", kl_weight=1.0)
        batch = _make_batch()
        torch.manual_seed(0)
        mod.training_step(batch, 0)
        logged = {
            c.args[0]: torch.as_tensor(c.args[1]).detach()
            for c in cast(MagicMock, mod.log).call_args_list
        }
        kl = float(logged["train_kl"])
        assert kl > 0
        assert float(logged["train_loss"]) == pytest.approx(
            float(logged["train_ce"]) + kl, rel=1e-4
        )

    def test_use_kl_false_trains_on_pure_ce(self):
        mod = _make_module(use_kl=False)
        batch = _make_batch()
        torch.manual_seed(0)
        mod.training_step(batch, 0)
        logged = {
            c.args[0]: torch.as_tensor(c.args[1]).detach()
            for c in cast(MagicMock, mod.log).call_args_list
        }
        assert float(logged["train_loss"]) == pytest.approx(
            float(logged["train_ce"]), rel=1e-6
        )

    def test_loss_backpropagates(self):
        mod = _make_module(kl_anneal="constant", kl_weight=1.0)
        loss = mod.training_step(_make_batch(), 0)
        loss.backward()
        grad = mod.model.sheaf_learner.lin_scale.weight.grad
        assert grad is not None and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Ensembled evaluation
# ---------------------------------------------------------------------------


class TestEnsembledEvaluation:
    def test_training_mode_uses_single_pass(self):
        mod = _make_module(ensemble_size=3)
        mod.train()
        mod._forward(_make_batch())
        assert mod._last_ensemble is None

    def test_eval_mode_ensembles(self):
        mod = _make_module(ensemble_size=3)
        mod.eval()
        batch = _make_batch()
        assert batch.x is not None
        num_nodes = batch.x.size(0)
        logits = mod._forward(batch)
        assert mod._last_ensemble is not None
        assert mod._last_ensemble.shape == (3, num_nodes, 3)
        # Returned logits are log-probabilities of the averaged prediction.
        probs = logits.exp()
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(num_nodes),
            atol=1e-4,
            rtol=0.0,
        )
        torch.testing.assert_close(
            probs, mod._last_ensemble.mean(dim=0), atol=1e-4, rtol=1e-4
        )

    def test_single_sample_ensemble_falls_back(self):
        mod = _make_module(ensemble_size=1)
        mod.eval()
        mod._forward(_make_batch())
        assert mod._last_ensemble is None

    def test_eval_step_logs_uncertainty_metrics(self):
        mod = _make_module(ensemble_size=3)
        mod.eval()
        mod._eval_step(_make_batch(), "val_mask", "val")
        logged = [c.args[0] for c in cast(MagicMock, mod.log).call_args_list]
        assert {
            "val_loss",
            "val_acc",
            "val_pred_entropy",
            "val_epistemic_var",
            "val_mutual_info",
            "val_ece",
        } <= set(logged)
