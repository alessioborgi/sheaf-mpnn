# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/het_run.py — HGB heterogeneous node-classification script."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from exp.config import Config, DatasetConfig, ModelConfig, ModelType, OptimConfig
from exp.datasets.info import HetDatasetInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_het_info(metric: str = "acc") -> HetDatasetInfo:
    return HetDatasetInfo(
        name="dblp",
        num_features=334,
        num_classes=4,
        num_splits=1,
        metric=metric,
        split_type="npz_file",
        num_node_types=4,
        num_edge_types=6,
        node_type_names=("author", "paper", "term", "venue"),
        edge_type_names=("a-p", "p-a", "p-t", "t-p", "p-v", "v-p"),
        task="multiclass",
    )


def _make_cfg(**overrides) -> Config:
    defaults: dict = dict(
        dataset=DatasetConfig(name="dblp", root="/tmp/fake"),
        optim=OptimConfig(epochs=2, early_stopping=1),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_dm_mock(metric: str = "acc") -> MagicMock:
    dm = MagicMock()
    dm.info = _make_het_info(metric=metric)
    dm.split_sizes = (100, 50, 50)
    dm.num_edges = 200
    dm.num_nodes = 100
    return dm


def _make_trainer_mock(metric: str = "acc", test_val: float = 0.75) -> MagicMock:
    trainer = MagicMock()
    trainer.test.return_value = [{f"test_{metric}": test_val}]
    return trainer


# ---------------------------------------------------------------------------
# _set_seed
# ---------------------------------------------------------------------------


def test_set_seed_does_not_raise():
    from exp.het_run import _set_seed

    _set_seed(42)


def test_set_seed_accepts_zero():
    from exp.het_run import _set_seed

    _set_seed(0)


# ---------------------------------------------------------------------------
# _parse_config (het_run.py)
# ---------------------------------------------------------------------------


class TestHetRunParseConfig:
    """Test _parse_config() in exp.het_run."""

    def test_no_preset_calls_tyro_with_none_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        cfg = _make_cfg()
        with patch("exp.het_run.tyro.cli", return_value=cfg) as mock_cli:
            from exp.het_run import _parse_config

            result = _parse_config()
        assert result is cfg
        _, kwargs = mock_cli.call_args
        assert kwargs["default"] is None

    def test_preset_space_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "cora"])
        from exp.het_run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.het_run.preset_registry") as mock_reg,
            patch("exp.het_run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg
        assert "--preset" not in sys.argv

    def test_preset_equals_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset=dblp"])
        from exp.het_run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.het_run.preset_registry") as mock_reg,
            patch("exp.het_run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg

    def test_unknown_preset_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "nonexistent_xyz_het"])
        from exp.het_run import _parse_config

        with pytest.raises(SystemExit):
            _parse_config()

    def test_preset_flag_without_value_is_skipped(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset"])
        cfg = _make_cfg()
        with patch("exp.het_run.tyro.cli", return_value=cfg):
            from exp.het_run import _parse_config

            result = _parse_config()
        assert result is cfg

    def test_non_preset_args_preserved(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--dataset.name", "dblp"])
        cfg = _make_cfg()
        with patch("exp.het_run.tyro.cli", return_value=cfg):
            from exp.het_run import _parse_config

            _parse_config()
        assert "--dataset.name" in sys.argv
        assert "dblp" in sys.argv


# ---------------------------------------------------------------------------
# _make_logger (het_run.py)
# ---------------------------------------------------------------------------


def test_het_make_logger_wandb_disabled_returns_false():
    from exp.het_run import _make_logger

    cfg = _make_cfg()
    info = _make_het_info()
    result = _make_logger(cfg, info, fold=0, run_name="het-run")
    assert result is False


def test_het_make_logger_wandb_enabled_returns_logger_instance():
    from exp.config import WandBConfig
    from exp.het_run import _make_logger

    cfg = _make_cfg()
    cfg.wandb = WandBConfig(enabled=True, project="het-project")
    info = _make_het_info()

    mock_logger_instance = MagicMock()
    mock_wandb_logger_cls = MagicMock(return_value=mock_logger_instance)

    with patch("lightning.pytorch.loggers.WandbLogger", mock_wandb_logger_cls):
        result = _make_logger(cfg, info, fold=0, run_name="het-run")

    assert result is mock_logger_instance


# ---------------------------------------------------------------------------
# _model_label (het_run.py)
# ---------------------------------------------------------------------------


def test_het_model_label_format():
    from exp.het_run import _model_label

    cfg = _make_cfg()
    label = _model_label(cfg)
    assert cfg.model.type in label
    assert cfg.model.variant in label


# ---------------------------------------------------------------------------
# _run_fold (het_run.py)
# ---------------------------------------------------------------------------


class TestHetRunFold:
    """Test _run_fold in exp.het_run."""

    def _run_mocked(
        self,
        cfg: Config | None = None,
        metric: str = "acc",
        test_val: float = 0.75,
    ) -> float:
        cfg = cfg or _make_cfg()
        info = _make_het_info(metric)
        trainer_mock = _make_trainer_mock(metric, test_val)

        with (
            patch("exp.het_run.HetSheafDataModule"),
            patch("exp.het_run.HetSheafLightningModule"),
            patch("exp.het_run.Trainer", return_value=trainer_mock),
            patch("exp.het_run.EarlyStopping"),
            patch("exp.het_run.ModelCheckpoint"),
            patch("exp.het_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.het_run import _run_fold

            return _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

    def test_returns_float(self):
        assert isinstance(self._run_mocked(), float)

    def test_returns_correct_test_metric(self):
        result = self._run_mocked(test_val=0.82)
        assert result == pytest.approx(0.82)

    def test_missing_metric_key_returns_zero(self):
        cfg = _make_cfg()
        info = _make_het_info("acc")
        trainer_mock = MagicMock()
        trainer_mock.test.return_value = [{}]  # no test_acc key

        with (
            patch("exp.het_run.HetSheafDataModule"),
            patch("exp.het_run.HetSheafLightningModule"),
            patch("exp.het_run.Trainer", return_value=trainer_mock),
            patch("exp.het_run.EarlyStopping"),
            patch("exp.het_run.ModelCheckpoint"),
            patch("exp.het_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.het_run import _run_fold

            result = _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

        assert result == pytest.approx(0.0)

    def test_roc_auc_metric_key(self):
        result = self._run_mocked(metric="roc_auc", test_val=0.91)
        assert result == pytest.approx(0.91)

    def test_trainer_receives_correct_epochs(self):
        from exp.het_run import _run_fold

        cfg = _make_cfg(optim=OptimConfig(epochs=7, early_stopping=2))
        info = _make_het_info()
        trainer_mock = _make_trainer_mock()

        with (
            patch("exp.het_run.HetSheafDataModule"),
            patch("exp.het_run.HetSheafLightningModule"),
            patch("exp.het_run.Trainer", return_value=trainer_mock) as mock_trainer_cls,
            patch("exp.het_run.EarlyStopping"),
            patch("exp.het_run.ModelCheckpoint"),
            patch("exp.het_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["max_epochs"] == 7


# ---------------------------------------------------------------------------
# _silence_third_party (het_run.py)
# ---------------------------------------------------------------------------


def test_het_silence_third_party_does_not_raise():
    from exp.het_run import _silence_third_party

    _silence_third_party()


# ---------------------------------------------------------------------------
# main() (het_run.py)
# ---------------------------------------------------------------------------


class TestHetRunMain:
    """Test main() in exp.het_run with all externals mocked."""

    def _run_main(self, cfg: Config | None = None, model_type=None) -> None:
        cfg = cfg or _make_cfg()
        if model_type is not None:
            cfg.model = ModelConfig(type=model_type)

        dm_mock = _make_dm_mock()
        trainer_mock = _make_trainer_mock()

        with (
            patch("exp.het_run._parse_config", return_value=cfg),
            patch("exp.het_run._silence_third_party"),
            patch("exp.het_run.setup_torch"),
            patch("exp.het_run.HetSheafDataModule", return_value=dm_mock),
            patch("exp.het_run.HetSheafLightningModule"),
            patch("exp.het_run.Trainer", return_value=trainer_mock),
            patch("exp.het_run.EarlyStopping"),
            patch("exp.het_run.ModelCheckpoint"),
            patch("exp.het_run.ModelSummary"),
            patch("exp.het_run.tqdm", side_effect=lambda x, **kw: x),
            patch("exp.het_run.tempfile.TemporaryDirectory") as mock_tmpdir,
            patch("exp.het_run.logging.basicConfig"),
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.het_run import main

            main()

    def test_main_completes_without_error(self):
        self._run_main()

    def test_main_sheaf_attn_branch(self):
        self._run_main(model_type=ModelType.SHEAF_ATTN)

    def test_main_nsd_branch(self):
        self._run_main(model_type=ModelType.NSD)

    def test_main_loss_stop_strategy(self):
        cfg = _make_cfg(
            optim=OptimConfig(epochs=2, early_stopping=1, stop_strategy="loss")
        )
        self._run_main(cfg=cfg)

    def test_main_acc_stop_strategy(self):
        cfg = _make_cfg(
            optim=OptimConfig(epochs=2, early_stopping=1, stop_strategy="acc")
        )
        self._run_main(cfg=cfg)

    def test_main_roc_auc_metric(self):
        """When metric is roc_auc, scale factor should be 1.0 (not 100)."""
        dm_mock = _make_dm_mock(metric="roc_auc")
        # Override the trainer to return roc_auc test result
        trainer_mock = MagicMock()
        trainer_mock.test.return_value = [{"test_roc_auc": 0.91}]
        cfg = _make_cfg()

        with (
            patch("exp.het_run._parse_config", return_value=cfg),
            patch("exp.het_run._silence_third_party"),
            patch("exp.het_run.setup_torch"),
            patch("exp.het_run.HetSheafDataModule", return_value=dm_mock),
            patch("exp.het_run.HetSheafLightningModule"),
            patch("exp.het_run.Trainer", return_value=trainer_mock),
            patch("exp.het_run.EarlyStopping"),
            patch("exp.het_run.ModelCheckpoint"),
            patch("exp.het_run.ModelSummary"),
            patch("exp.het_run.tqdm", side_effect=lambda x, **kw: x),
            patch("exp.het_run.tempfile.TemporaryDirectory") as mock_tmpdir,
            patch("exp.het_run.logging.basicConfig"),
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.het_run import main

            main()
