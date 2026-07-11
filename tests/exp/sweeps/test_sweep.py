# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/sweeps/sweep.py - _suggest, _build_cfg, _run_trial, main, E2E."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest
import torch
import yaml
from lightning import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from exp.config import Config, DatasetConfig, ModelType, OptimConfig
from exp.data import DatasetInfo
from exp.sweeps.models import (
    CategoricalParam,
    FloatParam,
    IntParam,
    SweepConfig,
)
from exp.sweeps.sweep import _build_cfg, _make_wandb_callbacks, _run_trial, _suggest

_PRESET_PATCH = "exp.sweeps.sweep.preset_registry.get_or_default"

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_base_cfg() -> Config:
    return Config(
        dataset=DatasetConfig(name="cora", root="/tmp/fake"),
        optim=OptimConfig(epochs=3, early_stopping=2),
    )


def _make_sweep_cfg(**optuna_overrides) -> SweepConfig:
    """Minimal sweep config with one param of each type."""
    return SweepConfig.model_validate(
        {
            "model": "nsd",
            "search_space": {
                "variant": {"type": "categorical", "choices": ["general", "diagonal"]},
                "stalk_dim": {"type": "int", "low": 2, "high": 4},
                "alpha": {"type": "float", "low": 0.1, "high": 2.0},
            },
            "config": {"n_trials": 1, "seed": 42, **optuna_overrides},
        }
    )


def _make_trial(params: dict | None = None) -> MagicMock:
    p = {"variant": "general", "stalk_dim": 3, "alpha": 1.0, **(params or {})}
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_categorical.side_effect = lambda n, choices, **kw: p[n]
    trial.suggest_int.side_effect = lambda n, lo, hi, **kw: p[n]
    trial.suggest_float.side_effect = lambda n, lo, hi, **kw: p[n]
    return trial


def _make_dm_mock(metric: str = "acc") -> MagicMock:
    dm = MagicMock()
    dm.info = DatasetInfo(
        name="cora",
        num_features=5,
        num_classes=3,
        num_splits=10,
        metric=metric,
        split_type="npz_file",
    )
    return dm


def _run_mocked(
    sweep_cfg: SweepConfig,
    base_cfg: Config,
    trial: MagicMock | None = None,
    *,
    val_metric: float = 0.75,
    metric: str = "acc",
) -> float:
    """Call _run_trial with all I/O mocked out."""
    if trial is None:
        trial = _make_trial()
    dm_mock = _make_dm_mock(metric=metric)
    trainer_mock = MagicMock()
    trainer_mock.current_epoch = 10
    trainer_mock.validate.return_value = [{f"val_{metric}": val_metric}]
    trainer_mock.test.return_value = [{f"test_{metric}": val_metric - 0.05}]
    with (
        patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
        patch("exp.sweeps.sweep.SheafLightningModule"),
        patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
        patch("exp.sweeps.sweep.EarlyStopping"),
    ):
        return _run_trial(trial, sweep_cfg, base_cfg)


# ---------------------------------------------------------------------------
# _suggest
# ---------------------------------------------------------------------------


class TestSuggest:
    def test_float_param_calls_suggest_float(self):
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.5
        spec = FloatParam(type="float", low=0.0, high=1.0)
        _suggest(trial, "lr", spec)
        trial.suggest_float.assert_called_once_with("lr", 0.0, 1.0, log=False)

    def test_float_param_log_flag_is_forwarded(self):
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.01
        spec = FloatParam(type="float", low=1e-4, high=1e-1, log=True)
        _suggest(trial, "lr", spec)
        trial.suggest_float.assert_called_once_with("lr", 1e-4, 1e-1, log=True)

    def test_int_param_calls_suggest_int(self):
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 3
        spec = IntParam(type="int", low=2, high=8)
        _suggest(trial, "stalk_dim", spec)
        trial.suggest_int.assert_called_once_with("stalk_dim", 2, 8, log=False)

    def test_categorical_param_calls_suggest_categorical(self):
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = "general"
        spec = CategoricalParam(type="categorical", choices=["general", "diagonal"])
        _suggest(trial, "variant", spec)
        trial.suggest_categorical.assert_called_once_with(
            "variant", ["general", "diagonal"]
        )

    def test_returns_value_from_trial(self):
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.42
        spec = FloatParam(type="float", low=0.0, high=1.0)
        assert _suggest(trial, "x", spec) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# _build_cfg
# ---------------------------------------------------------------------------


class TestBuildCfg:
    def test_model_param_routed_to_model_config(self):
        cfg = _build_cfg(_make_base_cfg(), "nsd", {"stalk_dim": 6})
        assert cfg.model.stalk_dim == 6

    def test_reg_param_routed_to_reg_config(self):
        cfg = _build_cfg(_make_base_cfg(), "nsd", {"input_dropout": 0.4})
        assert cfg.reg.input_dropout == pytest.approx(0.4)

    def test_optim_param_routed_to_optim_config(self):
        cfg = _build_cfg(_make_base_cfg(), "nsd", {"lr": 0.001})
        assert cfg.optim.lr == pytest.approx(0.001)

    def test_model_type_is_set_from_string(self):
        cfg = _build_cfg(_make_base_cfg(), "nsd", {})
        assert cfg.model.type == ModelType.NSD

    def test_mixed_params_routed_correctly(self):
        cfg = _build_cfg(
            _make_base_cfg(),
            "nsd",
            {"stalk_dim": 3, "input_dropout": 0.2, "lr": 0.005},
        )
        assert cfg.model.stalk_dim == 3
        assert cfg.model.type == ModelType.NSD
        assert cfg.reg.input_dropout == pytest.approx(0.2)
        assert cfg.optim.lr == pytest.approx(0.005)

    def test_base_cfg_is_not_mutated(self):
        base = _make_base_cfg()
        original_stalk = base.model.stalk_dim
        _build_cfg(base, "nsd", {"stalk_dim": 99})
        assert base.model.stalk_dim == original_stalk

    def test_unknown_param_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown sweep parameter 'not_a_field'"):
            _build_cfg(_make_base_cfg(), "nsd", {"not_a_field": 42})


# ---------------------------------------------------------------------------
# _run_trial
# ---------------------------------------------------------------------------


class TestRunTrial:
    @pytest.fixture
    def sweep_cfg(self):
        return _make_sweep_cfg()

    @pytest.fixture
    def base_cfg(self):
        return _make_base_cfg()

    def test_returns_float(self, sweep_cfg, base_cfg):
        assert isinstance(_run_mocked(sweep_cfg, base_cfg), float)

    def test_single_seed_no_penalty_equals_val_metric(self, base_cfg):
        sweep_cfg = _make_sweep_cfg(n_seeds_per_trial=1, std_weight=0.0)
        result = _run_mocked(sweep_cfg, base_cfg, val_metric=0.82)
        assert result == pytest.approx(0.82)

    def test_std_weight_penalises_variance(self, base_cfg):
        sweep_cfg = _make_sweep_cfg(n_seeds_per_trial=2, std_weight=1.0)
        trial = _make_trial()
        dm_mock = _make_dm_mock()
        trainer_mock = MagicMock()
        trainer_mock.validate.side_effect = [
            [{"val_acc": 0.60}],
            [{"val_acc": 0.80}],
        ]
        with (
            patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
            patch("exp.sweeps.sweep.SheafLightningModule"),
            patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
            patch("exp.sweeps.sweep.EarlyStopping"),
        ):
            result = _run_trial(trial, sweep_cfg, base_cfg)
        expected = float(np.mean([0.60, 0.80])) - 1.0 * float(np.std([0.60, 0.80]))
        assert result == pytest.approx(expected)

    def test_sets_val_mean_user_attr(self, sweep_cfg, base_cfg):
        trial = _make_trial()
        _run_mocked(sweep_cfg, base_cfg, trial=trial, val_metric=0.77)
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert attrs["val_mean"] == pytest.approx(0.77)

    def test_sets_val_std_zero_for_single_seed(self, sweep_cfg, base_cfg):
        trial = _make_trial()
        _run_mocked(sweep_cfg, base_cfg, trial=trial)
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert attrs["val_std"] == pytest.approx(0.0)

    def test_sets_n_seeds_user_attr(self, base_cfg):
        sweep_cfg = _make_sweep_cfg(n_seeds_per_trial=3)
        trial = _make_trial()
        dm_mock = _make_dm_mock()
        trainer_mock = MagicMock()
        trainer_mock.validate.return_value = [{"val_acc": 0.75}]
        with (
            patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
            patch("exp.sweeps.sweep.SheafLightningModule"),
            patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
            patch("exp.sweeps.sweep.EarlyStopping"),
        ):
            _run_trial(trial, sweep_cfg, base_cfg)
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert attrs["n_seeds"] == 3

    def test_trainer_fit_called_once_per_seed(self, base_cfg):
        sweep_cfg = _make_sweep_cfg(n_seeds_per_trial=4)
        dm_mock = _make_dm_mock()
        trainer_mock = MagicMock()
        trainer_mock.validate.return_value = [{"val_acc": 0.75}]
        with (
            patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
            patch("exp.sweeps.sweep.SheafLightningModule"),
            patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
            patch("exp.sweeps.sweep.EarlyStopping"),
        ):
            _run_trial(_make_trial(), sweep_cfg, base_cfg)
        assert trainer_mock.fit.call_count == 4

    def test_missing_val_key_defaults_to_zero(self, sweep_cfg, base_cfg):
        dm_mock = _make_dm_mock(metric="acc")
        trainer_mock = MagicMock()
        trainer_mock.validate.return_value = [{}]  # no val_acc key
        with (
            patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
            patch("exp.sweeps.sweep.SheafLightningModule"),
            patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
            patch("exp.sweeps.sweep.EarlyStopping"),
        ):
            result = _run_trial(_make_trial(), sweep_cfg, base_cfg)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _make_wandb_callbacks
# ---------------------------------------------------------------------------


class TestMakeWandbCallbacks:
    def test_missing_integration_returns_empty_list(self):
        sweep_cfg = _make_sweep_cfg()
        sweep_cfg = SweepConfig.model_validate(
            {
                **sweep_cfg.model_dump(),
                "config": {
                    **sweep_cfg.config.model_dump(),
                    "wandb_project": "my-project",
                },
            }
        )
        with patch.dict(
            sys.modules,
            {"optuna_integration": None, "optuna_integration.wandb": None},
        ):
            result = _make_wandb_callbacks(_make_base_cfg(), sweep_cfg)
        assert result == []

    def test_missing_integration_prints_warning(self, capsys):
        sweep_cfg = SweepConfig.model_validate(
            {
                "model": "nsd",
                "search_space": {},
                "config": {"wandb_project": "p"},
            }
        )
        with patch.dict(
            sys.modules,
            {"optuna_integration": None, "optuna_integration.wandb": None},
        ):
            _make_wandb_callbacks(_make_base_cfg(), sweep_cfg)
        assert "not installed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    @pytest.fixture(autouse=True)
    def _chdir_to_tmp(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

    def _make_yaml_file(self, tmp_path: Path, **overrides) -> Path:
        content = {
            "model": "nsd",
            "search_space": {
                "stalk_dim": {"type": "int", "low": 2, "high": 4},
            },
            "config": {"n_trials": 2, "seed": 0},
            **overrides,
        }
        p = tmp_path / "sweep.yaml"
        p.write_text(yaml.dump(content))
        return p

    def _make_study_mock(self) -> MagicMock:
        study = MagicMock()
        study.best_trial.number = 0
        study.best_trial.value = 0.75
        study.best_trial.user_attrs = {
            "val_mean": 0.75,
            "val_std": 0.0,
            "n_seeds": 1,
        }
        study.best_trial.params = {"stalk_dim": 3}
        return study

    def test_main_creates_study_and_calls_optimize(self, tmp_path):
        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(tmp_path)
        study_mock = self._make_study_mock()
        with (
            patch(_PRESET_PATCH, return_value=Config()),
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch("exp.sweeps.sweep.optuna.create_study", return_value=study_mock),
        ):
            main(yaml_path=yaml_path)
        study_mock.optimize.assert_called_once()

    def test_main_caps_total_trials_via_callback(self, tmp_path):
        from optuna.study import MaxTrialsCallback

        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(tmp_path)
        study_mock = self._make_study_mock()
        with (
            patch(_PRESET_PATCH, return_value=Config()),
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch("exp.sweeps.sweep.optuna.create_study", return_value=study_mock),
        ):
            main(yaml_path=yaml_path)
        _, kwargs = study_mock.optimize.call_args
        # The cap must be a study-wide callback, not a per-invocation n_trials,
        # so resumed or distributed sweeps stop at the configured total.
        assert "n_trials" not in kwargs
        max_trials_cbs = [
            cb for cb in kwargs["callbacks"] if isinstance(cb, MaxTrialsCallback)
        ]
        assert len(max_trials_cbs) == 1
        assert max_trials_cbs[0]._n_trials == 2

    def test_main_creates_maximize_study(self, tmp_path):
        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(tmp_path)
        study_mock = self._make_study_mock()
        with (
            patch(_PRESET_PATCH, return_value=Config()),
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch(
                "exp.sweeps.sweep.optuna.create_study", return_value=study_mock
            ) as create_mock,
        ):
            main(yaml_path=yaml_path)
        assert create_mock.call_args.kwargs["direction"] == "maximize"

    def test_main_prints_best_trial(self, tmp_path, capsys):
        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(tmp_path)
        study_mock = self._make_study_mock()
        with (
            patch(_PRESET_PATCH, return_value=Config()),
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch("exp.sweeps.sweep.optuna.create_study", return_value=study_mock),
        ):
            main(yaml_path=yaml_path)
        out = capsys.readouterr().out
        assert "Best trial" in out
        assert "stalk_dim" in out

    def test_dataset_yaml_overrides_preset(self, tmp_path):
        """Dataset block in YAML must replace the preset's dataset.name."""
        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(
            tmp_path, dataset={"name": "texas", "root": "exp/data"}
        )
        study_mock = self._make_study_mock()
        captured: list[Config] = []

        with (
            patch(_PRESET_PATCH, return_value=Config()),  # default: name="cora"
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch("exp.sweeps.sweep.optuna.create_study", return_value=study_mock),
        ):
            main(yaml_path=yaml_path)

        # Call the lambda passed to study.optimize with _run_trial patched so we
        # can inspect the base_cfg it receives.
        objective = study_mock.optimize.call_args[0][0]
        with patch(
            "exp.sweeps.sweep._run_trial",
            side_effect=lambda t, sc, bc: captured.append(bc) or 0.75,
        ):
            objective(_make_trial())

        assert captured[0].dataset.name == "texas"

    def test_main_prints_objective_line_when_std_weight_positive(
        self, tmp_path, capsys
    ):
        """Line 307: objective line is printed only when std_weight > 0."""
        from exp.sweeps.sweep import main

        yaml_path = self._make_yaml_file(
            tmp_path, config={"n_trials": 1, "seed": 0, "std_weight": 0.5}
        )
        study_mock = self._make_study_mock()
        # Give the study a non-zero value so the objective line is meaningful
        study_mock.best_trial.value = 0.70
        study_mock.best_trial.user_attrs = {
            "val_mean": 0.75,
            "val_std": 0.1,
            "n_seeds": 2,
        }
        with (
            patch(_PRESET_PATCH, return_value=Config()),
            patch("exp.sweeps.sweep.setup_torch"),
            patch(
                "exp.sweeps.sweep._run_final_test",
                return_value={"mean_test_acc": 0.8, "std_test_acc": 0.01},
            ),
            patch("exp.sweeps.sweep.optuna.create_study", return_value=study_mock),
        ):
            main(yaml_path=yaml_path)
        out = capsys.readouterr().out
        assert "objective" in out, (
            "Expected 'objective' line in output for std_weight > 0"
        )
        assert "0.5" in out or "0.50" in out, (
            "Expected std_weight printed in objective line"
        )


# ---------------------------------------------------------------------------
# E2E integration test  real Trainer + toy in-memory data
# ---------------------------------------------------------------------------


class _ToyDataModule(LightningDataModule):
    """Minimal in-memory data module for E2E testing without real datasets."""

    def __init__(self):
        super().__init__()
        self.info = DatasetInfo(
            name="toy",
            num_features=4,
            num_classes=3,
            num_splits=10,
            metric="acc",
            split_type="random",
        )

    def setup(self, stage=None):
        torch.manual_seed(0)
        n = 20
        x = torch.randn(n, 4)
        edge_index = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
        y = torch.randint(0, 3, (n,))
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:12] = True
        val_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[12:16] = True
        test_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[16:] = True
        self._data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def train_dataloader(self):
        return PyGDataLoader([self._data], batch_size=1)

    def val_dataloader(self):
        return PyGDataLoader([self._data], batch_size=1)

    def test_dataloader(self):
        return PyGDataLoader([self._data], batch_size=1)


class TestE2E:
    def test_single_trial_with_real_trainer_and_module(self):
        """Run one trial end-to-end: real SheafLightningModule + Trainer, toy data."""
        sweep_cfg = SweepConfig.model_validate(
            {
                "model": "nsd",
                "search_space": {
                    "variant": {"type": "categorical", "choices": ["general"]},
                    "stalk_dim": {"type": "int", "low": 2, "high": 2},
                    "hidden_dim": {"type": "int", "low": 4, "high": 4},
                    "num_layers": {"type": "int", "low": 1, "high": 1},
                },
                "config": {"n_trials": 1, "n_seeds_per_trial": 1, "seed": 0},
            }
        )
        base_cfg = Config(optim=OptimConfig(epochs=2, early_stopping=1))

        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = lambda n, choices, **kw: choices[0]
        trial.suggest_int.side_effect = lambda n, lo, hi, **kw: lo
        trial.suggest_float.side_effect = lambda n, lo, hi, **kw: lo

        toy_dm = _ToyDataModule()

        with (
            patch("exp.sweeps.sweep.SheafDataModule", return_value=toy_dm),
            patch("exp.sweeps.sweep._PruningCb", None),
        ):
            result = _run_trial(trial, sweep_cfg, base_cfg)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert "val_mean" in attrs
        assert "val_std" in attrs


class TestPerTrialTestMetrics:
    """Every trial must record test metrics at the best-val checkpoint."""

    def test_test_metrics_aggregated_with_std(self, tmp_path):
        trial = _make_trial()
        sweep_cfg = _make_sweep_cfg()
        base_cfg = _make_base_cfg()
        _run_mocked(sweep_cfg, base_cfg, trial=trial)
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert attrs["mean_test_acc"] == pytest.approx(0.70)
        assert attrs["std_test_acc"] == 0.0
        assert attrs["mean_val_acc"] == pytest.approx(0.75)
        assert attrs["std_val_acc"] == 0.0

    def test_resource_metrics_aggregated(self, tmp_path):
        trial = _make_trial()
        _run_mocked(_make_sweep_cfg(), _make_base_cfg(), trial=trial)
        attrs = {c.args[0]: c.args[1] for c in trial.set_user_attr.call_args_list}
        assert attrs["mean_fit_time_s"] > 0.0
        assert attrs["std_fit_time_s"] == 0.0
        # The mocked trainer reports 10 epochs, so per-epoch = total / 10.
        assert attrs["mean_epoch_time_s"] == pytest.approx(
            attrs["mean_fit_time_s"] / 10
        )
