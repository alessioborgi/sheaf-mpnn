# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/gc_run.py — graph-classification training script."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from exp.config import Config, DatasetConfig, ModelConfig, ModelType, OptimConfig
from exp.datasets.info import GCDatasetInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gc_info(metric: str = "acc") -> GCDatasetInfo:
    return GCDatasetInfo(
        name="mutag",
        num_features=7,
        num_classes=2,
        num_splits=10,
        metric=metric,
        split_type="seed",
        num_node_types=2,
        num_edge_types=4,
        task="multiclass",
    )


def _make_cfg(**overrides) -> Config:
    defaults: dict = dict(
        dataset=DatasetConfig(name="mutag", root="/tmp/fake"),
        optim=OptimConfig(epochs=2, early_stopping=1),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_dm_mock(metric: str = "acc", num_splits: int = 3) -> MagicMock:
    dm = MagicMock()
    dm.info = GCDatasetInfo(
        name="mutag",
        num_features=7,
        num_classes=2,
        num_splits=num_splits,
        metric=metric,
        split_type="seed",
        num_node_types=2,
        num_edge_types=4,
        task="multiclass",
    )
    dm.split_sizes = (80, 20, 20)
    dm.num_edges = 100
    dm.num_nodes = 50
    return dm


def _make_trainer_mock(test_val: float = 0.75) -> MagicMock:
    trainer = MagicMock()
    trainer.test.return_value = [{"test_acc": test_val}]
    return trainer


# ---------------------------------------------------------------------------
# _set_seed
# ---------------------------------------------------------------------------


def test_set_seed_does_not_raise():
    from exp.gc_run import _set_seed

    _set_seed(42)


def test_set_seed_accepts_zero():
    from exp.gc_run import _set_seed

    _set_seed(0)


# ---------------------------------------------------------------------------
# _make_dm
# ---------------------------------------------------------------------------


def test_make_dm_het_dataset_returns_gchet_module():
    from exp.gc_run import _make_dm

    with (
        patch("exp.gc_run.GCHetSheafDataModule") as MockHet,
        patch("exp.gc_run.GCSheafDataModule") as MockHomo,
    ):
        MockHet.return_value = MagicMock()
        _make_dm("mutag", root="/tmp", fold=0)
    MockHet.assert_called_once()
    MockHomo.assert_not_called()


def test_make_dm_homo_dataset_returns_gc_module():
    from exp.gc_run import _make_dm

    with (
        patch("exp.gc_run.GCHetSheafDataModule") as MockHet,
        patch("exp.gc_run.GCSheafDataModule") as MockHomo,
    ):
        MockHomo.return_value = MagicMock()
        _make_dm("collab", root="/tmp", fold=0)
    MockHomo.assert_called_once()
    MockHet.assert_not_called()


def test_make_dm_all_keyword_args_forwarded():
    from exp.gc_run import _make_dm

    with patch("exp.gc_run.GCSheafDataModule") as MockHomo:
        MockHomo.return_value = MagicMock()
        _make_dm(
            "collab",
            root="/data",
            fold=2,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )
    _, kwargs = MockHomo.call_args
    assert kwargs["fold"] == 2
    assert kwargs["batch_size"] == 32
    assert kwargs["num_workers"] == 4


def test_make_dm_name_case_insensitive():
    """Het registry lookup is case-insensitive (name.lower())."""
    from exp.gc_run import _make_dm

    with (
        patch("exp.gc_run.GCHetSheafDataModule") as MockHet,
        patch("exp.gc_run.GCSheafDataModule") as MockHomo,
    ):
        MockHet.return_value = MagicMock()
        _make_dm("MUTAG", root="/tmp", fold=0)
    MockHet.assert_called_once()
    MockHomo.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_config (gc_run.py)
# ---------------------------------------------------------------------------


class TestGcRunParseConfig:
    """Test _parse_config() in exp.gc_run."""

    def test_no_preset_calls_tyro_with_none_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        cfg = _make_cfg()
        with patch("exp.gc_run.tyro.cli", return_value=cfg) as mock_cli:
            from exp.gc_run import _parse_config

            result = _parse_config()
        assert result is cfg
        _, kwargs = mock_cli.call_args
        assert kwargs["default"] is None

    def test_preset_space_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "cora"])
        from exp.gc_run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.gc_run.preset_registry") as mock_reg,
            patch("exp.gc_run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg
        assert "--preset" not in sys.argv

    def test_preset_equals_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset=cora"])
        from exp.gc_run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.gc_run.preset_registry") as mock_reg,
            patch("exp.gc_run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg

    def test_unknown_preset_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "nonexistent_xyz_gc"])
        from exp.gc_run import _parse_config

        with pytest.raises(SystemExit):
            _parse_config()

    def test_preset_flag_without_value_is_skipped(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset"])
        cfg = _make_cfg()
        with patch("exp.gc_run.tyro.cli", return_value=cfg):
            from exp.gc_run import _parse_config

            result = _parse_config()
        assert result is cfg

    def test_non_preset_args_preserved(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--dataset.name", "mutag"])
        cfg = _make_cfg()
        with patch("exp.gc_run.tyro.cli", return_value=cfg):
            from exp.gc_run import _parse_config

            _parse_config()
        assert "--dataset.name" in sys.argv
        assert "mutag" in sys.argv


# ---------------------------------------------------------------------------
# _make_logger (gc_run.py)
# ---------------------------------------------------------------------------


def test_gc_make_logger_wandb_disabled_returns_false():
    from exp.gc_run import _make_logger

    cfg = _make_cfg()
    info = _make_gc_info()
    result = _make_logger(cfg, info, fold=0, run_name="gc-run")
    assert result is False


def test_gc_make_logger_wandb_enabled_returns_logger_instance():
    from exp.config import WandBConfig
    from exp.gc_run import _make_logger

    cfg = _make_cfg()
    cfg.wandb = WandBConfig(enabled=True, project="gc-project")
    info = _make_gc_info()

    mock_logger_instance = MagicMock()
    mock_wandb_logger_cls = MagicMock(return_value=mock_logger_instance)

    with patch("lightning.pytorch.loggers.WandbLogger", mock_wandb_logger_cls):
        result = _make_logger(cfg, info, fold=0, run_name="gc-run")

    assert result is mock_logger_instance


# ---------------------------------------------------------------------------
# _model_label (gc_run.py)
# ---------------------------------------------------------------------------


def test_gc_model_label_format():
    from exp.gc_run import _model_label

    cfg = _make_cfg()
    label = _model_label(cfg)
    # Should be "<type>-<variant>"
    assert cfg.model.type in label
    assert cfg.model.variant in label


# ---------------------------------------------------------------------------
# _run_fold (gc_run.py)
# ---------------------------------------------------------------------------


class TestGcRunFold:
    """Test _run_fold in exp.gc_run."""

    def _run_mocked(
        self,
        cfg: Config | None = None,
        test_val: float = 0.75,
    ) -> float:
        cfg = cfg or _make_cfg()
        info = _make_gc_info()
        trainer_mock = _make_trainer_mock(test_val)

        with (
            patch("exp.gc_run._make_dm") as mock_dm_fn,
            patch("exp.gc_run.GCLightningModule"),
            patch("exp.gc_run.Trainer", return_value=trainer_mock),
            patch("exp.gc_run.EarlyStopping"),
            patch("exp.gc_run.ModelCheckpoint"),
            patch("exp.gc_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            dm_mock = MagicMock()
            mock_dm_fn.return_value = dm_mock
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.gc_run import _run_fold

            return _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

    def test_returns_float(self):
        assert isinstance(self._run_mocked(), float)

    def test_returns_correct_test_acc(self):
        result = self._run_mocked(test_val=0.82)
        assert result == pytest.approx(0.82)

    def test_missing_test_acc_key_returns_zero(self):
        cfg = _make_cfg()
        info = _make_gc_info()
        trainer_mock = MagicMock()
        trainer_mock.test.return_value = [{}]  # no test_acc key

        with (
            patch("exp.gc_run._make_dm") as mock_dm_fn,
            patch("exp.gc_run.GCLightningModule"),
            patch("exp.gc_run.Trainer", return_value=trainer_mock),
            patch("exp.gc_run.EarlyStopping"),
            patch("exp.gc_run.ModelCheckpoint"),
            patch("exp.gc_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_dm_fn.return_value = MagicMock()
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.gc_run import _run_fold

            result = _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

        assert result == pytest.approx(0.0)

    def test_trainer_receives_correct_epochs(self):
        from exp.gc_run import _run_fold

        cfg = _make_cfg(optim=OptimConfig(epochs=5, early_stopping=2))
        info = _make_gc_info()
        trainer_mock = _make_trainer_mock()

        with (
            patch("exp.gc_run._make_dm") as mock_dm_fn,
            patch("exp.gc_run.GCLightningModule"),
            patch("exp.gc_run.Trainer", return_value=trainer_mock) as mock_trainer_cls,
            patch("exp.gc_run.EarlyStopping"),
            patch("exp.gc_run.ModelCheckpoint"),
            patch("exp.gc_run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_dm_fn.return_value = MagicMock()
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["max_epochs"] == 5


# ---------------------------------------------------------------------------
# _silence_third_party (gc_run.py)
# ---------------------------------------------------------------------------


def test_gc_silence_third_party_does_not_raise():
    from exp.gc_run import _silence_third_party

    _silence_third_party()


# ---------------------------------------------------------------------------
# main() (gc_run.py)
# ---------------------------------------------------------------------------


class TestGcRunMain:
    """Test main() in exp.gc_run with all externals mocked."""

    def _run_main(self, cfg: Config | None = None, model_type=None) -> None:
        cfg = cfg or _make_cfg()
        if model_type is not None:
            cfg.model = ModelConfig(type=model_type)

        dm_mock = _make_dm_mock(num_splits=1)
        trainer_mock = _make_trainer_mock()

        with (
            patch("exp.gc_run._parse_config", return_value=cfg),
            patch("exp.gc_run._silence_third_party"),
            patch("exp.gc_run.setup_torch"),
            patch("exp.gc_run._make_dm", return_value=dm_mock),
            patch("exp.gc_run.GCLightningModule"),
            patch("exp.gc_run.Trainer", return_value=trainer_mock),
            patch("exp.gc_run.EarlyStopping"),
            patch("exp.gc_run.ModelCheckpoint"),
            patch("exp.gc_run.ModelSummary"),
            patch("exp.gc_run.tqdm", side_effect=lambda x, **kw: x),
            patch("exp.gc_run.tempfile.TemporaryDirectory") as mock_tmpdir,
            patch("exp.gc_run.logging.basicConfig"),
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False

            from exp.gc_run import main

            main()

    def test_main_completes_without_error(self):
        self._run_main()

    def test_main_sheaf_attn_branch(self):
        self._run_main(model_type=ModelType.SHEAF_ATTN)

    def test_main_nsd_branch(self):
        self._run_main(model_type=ModelType.NSD)

    def test_main_uses_loss_stop_strategy(self):
        cfg = _make_cfg(
            optim=OptimConfig(epochs=2, early_stopping=1, stop_strategy="loss")
        )
        self._run_main(cfg=cfg)

    def test_main_uses_acc_stop_strategy(self):
        cfg = _make_cfg(
            optim=OptimConfig(epochs=2, early_stopping=1, stop_strategy="acc")
        )
        self._run_main(cfg=cfg)
