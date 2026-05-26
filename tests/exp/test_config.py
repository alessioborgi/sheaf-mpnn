# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Tests for exp/config.py -- typed configuration dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from exp.config import (
    Config,
    CVConfig,
    DatasetConfig,
    HardwareConfig,
    ModelConfig,
    OptimConfig,
    RegConfig,
    WandBConfig,
)


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.name == "cora"
        assert cfg.root.endswith("exp/data")

    def test_custom_values(self):
        cfg = DatasetConfig(name="texas", root="/tmp/data")
        assert cfg.name == "texas"
        assert cfg.root == "/tmp/data"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DatasetConfig)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.variant == "general"
        assert cfg.stalk_dim == 4
        assert cfg.hidden_dim == 16
        assert cfg.num_layers == 2
        assert cfg.alpha == 1.0

    @pytest.mark.parametrize("variant", ["diagonal", "general", "orthogonal"])
    def test_valid_variants(self, variant):
        cfg = ModelConfig(variant=variant)
        assert cfg.variant == variant

    def test_custom_dims(self):
        cfg = ModelConfig(stalk_dim=8, hidden_dim=64, num_layers=4, alpha=0.5)
        assert cfg.stalk_dim == 8
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 4
        assert cfg.alpha == pytest.approx(0.5)


class TestRegConfig:
    def test_defaults(self):
        cfg = RegConfig()
        assert cfg.input_dropout == 0.0
        assert cfg.dropout == 0.0

    def test_custom_dropout(self):
        cfg = RegConfig(input_dropout=0.5, dropout=0.3)
        assert cfg.input_dropout == pytest.approx(0.5)
        assert cfg.dropout == pytest.approx(0.3)


class TestOptimConfig:
    def test_defaults(self):
        cfg = OptimConfig()
        assert cfg.lr == pytest.approx(0.01)
        assert cfg.weight_decay == pytest.approx(5e-4)
        assert cfg.epochs == 1000
        assert cfg.early_stopping == 200
        assert cfg.stop_strategy == "loss"

    @pytest.mark.parametrize("strategy", ["loss", "acc"])
    def test_stop_strategies(self, strategy):
        cfg = OptimConfig(stop_strategy=strategy)
        assert cfg.stop_strategy == strategy

    def test_custom_lr_and_epochs(self):
        cfg = OptimConfig(lr=1e-3, epochs=500, early_stopping=50)
        assert cfg.lr == pytest.approx(1e-3)
        assert cfg.epochs == 500
        assert cfg.early_stopping == 50


class TestCVConfig:
    def test_defaults(self):
        cfg = CVConfig()
        assert cfg.folds == 10
        assert cfg.seed == 42
        assert cfg.min_acc == pytest.approx(0.0)

    def test_custom_folds_and_seed(self):
        cfg = CVConfig(folds=5, seed=0, min_acc=0.7)
        assert cfg.folds == 5
        assert cfg.seed == 0
        assert cfg.min_acc == pytest.approx(0.7)


class TestHardwareConfig:
    def test_defaults(self):
        cfg = HardwareConfig()
        assert cfg.cuda == 0

    def test_custom_cuda(self):
        cfg = HardwareConfig(cuda=1)
        assert cfg.cuda == 1


class TestWandBConfig:
    def test_defaults(self):
        cfg = WandBConfig()
        assert cfg.enabled is False
        assert cfg.entity is None
        assert cfg.project is None

    def test_enabled_with_project(self):
        cfg = WandBConfig(enabled=True, entity="myteam", project="nsd-bench")
        assert cfg.enabled is True
        assert cfg.entity == "myteam"
        assert cfg.project == "nsd-bench"


class TestConfig:
    def test_defaults_produce_correct_sub_configs(self):
        cfg = Config()
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.reg, RegConfig)
        assert isinstance(cfg.optim, OptimConfig)
        assert isinstance(cfg.cv, CVConfig)
        assert isinstance(cfg.hardware, HardwareConfig)
        assert isinstance(cfg.wandb, WandBConfig)

    def test_sub_config_defaults_are_independent_instances(self):
        cfg1 = Config()
        cfg2 = Config()
        assert cfg1.dataset is not cfg2.dataset
        assert cfg1.model is not cfg2.model

    def test_nested_overrides(self):
        cfg = Config(
            dataset=DatasetConfig(name="texas"),
            model=ModelConfig(stalk_dim=3, hidden_dim=16, num_layers=1),
        )
        assert cfg.dataset.name == "texas"
        assert cfg.model.stalk_dim == 3
        assert cfg.model.hidden_dim == 16
        # Unmodified sub-configs retain their defaults.
        assert cfg.optim.lr == pytest.approx(0.01)
        assert cfg.cv.folds == 10

    def test_dataclasses_replace(self):
        cfg = Config()
        new_cfg = dataclasses.replace(cfg, model=ModelConfig(stalk_dim=6))
        assert new_cfg.model.stalk_dim == 6
        assert new_cfg.dataset.name == "cora"  # unchanged
