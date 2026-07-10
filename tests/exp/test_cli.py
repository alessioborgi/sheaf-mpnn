# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the unified sheaf CLI and extracted run/splits logic."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from exp.config import Config, DatasetConfig, OptimConfig
from exp.data import DatasetInfo
from exp.gen_splits import (
    SplitsConfig,
    download_canonical_splits,
    generate_splits,
    splits,
)
from exp.run import _run_fold, run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_info(metric: str = "acc") -> DatasetInfo:
    return DatasetInfo(
        name="cora",
        num_features=5,
        num_classes=3,
        num_splits=10,
        metric=metric,
        split_type="npz_file",
    )


def _make_cfg(**overrides) -> Config:
    return Config(
        dataset=DatasetConfig(name="cora", root="/tmp/fake"),
        optim=OptimConfig(epochs=2, early_stopping=1),
        **overrides,
    )


def _make_dm_mock(metric: str = "acc", num_splits: int = 10) -> MagicMock:
    dm = MagicMock()
    dm.info = DatasetInfo(
        name="cora",
        num_features=5,
        num_classes=3,
        num_splits=num_splits,
        metric=metric,
        split_type="npz_file",
    )
    dm.split_sizes = (100, 50, 50)
    dm.num_edges = 200
    dm.num_nodes = 100
    dm.homophily = 0.8
    return dm


def _make_trainer_mock(metric: str = "acc", test_val: float = 0.75) -> MagicMock:
    trainer = MagicMock()
    trainer.test.return_value = [{f"test_{metric}": test_val}]
    return trainer


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCliDispatch:
    """Verify that cli.main() routes each subcommand to the right function."""

    def test_run_subcommand_calls_run(self, monkeypatch):
        called_with = []

        def fake_parse_config():
            return _make_cfg()

        def fake_run(cfg):
            called_with.append(cfg)
            return [0.75]

        monkeypatch.setattr(sys, "argv", ["sheaf", "run"])
        with (
            patch("exp.run._parse_config", side_effect=fake_parse_config),
            patch("exp.run.run", side_effect=fake_run),
        ):
            from exp import cli

            cli.main()

        assert len(called_with) == 1
        assert isinstance(called_with[0], Config)

    def test_splits_subcommand_calls_splits(self, monkeypatch):
        called_with = []

        monkeypatch.setattr(
            sys, "argv", ["sheaf", "splits", "--folds", "2", "--overwrite"]
        )
        with patch(
            "exp.gen_splits.splits", side_effect=lambda cfg: called_with.append(cfg)
        ):
            from exp import cli

            cli.main()

        assert len(called_with) == 1
        assert isinstance(called_with[0], SplitsConfig)
        assert called_with[0].folds == 2
        assert called_with[0].overwrite is True

    def test_sweep_subcommand_calls_sweep(self, monkeypatch, tmp_path):
        yaml_file = tmp_path / "sweep.yaml"
        yaml_file.write_text("model: nsd\nsearch_space: {}\nconfig: {n_trials: 1}\n")
        called_with = []

        monkeypatch.setattr(
            sys, "argv", ["sheaf", "sweep", "--yaml-path", str(yaml_file)]
        )
        with patch(
            "exp.sweeps.sweep.sweep",
            side_effect=lambda yaml_path, preset: called_with.append(
                (yaml_path, preset)
            ),
        ):
            from exp import cli

            cli.main()

        assert len(called_with) == 1
        assert called_with[0][0] == yaml_file
        assert called_with[0][1] is None

    def test_unknown_subcommand_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sheaf", "unknown"])
        from exp import cli

        with pytest.raises(SystemExit):
            cli.main()

    def test_no_args_exits_zero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sheaf"])
        from exp import cli

        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _run_fold
# ---------------------------------------------------------------------------


class TestRunFold:
    """Test the per-fold training helper in isolation."""

    def _run_mocked(
        self,
        cfg: Config | None = None,
        metric: str = "acc",
        test_val: float = 0.75,
    ) -> float:
        cfg = cfg or _make_cfg()
        info = _make_info(metric)
        trainer_mock = _make_trainer_mock(metric, test_val)

        with (
            patch("exp.run.SheafDataModule"),
            patch("exp.run.SheafLightningModule"),
            patch("exp.run.Trainer", return_value=trainer_mock),
            patch("exp.run.EarlyStopping"),
            patch("exp.run.ModelCheckpoint"),
            patch("exp.run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False
            return _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

    def test_returns_float(self):
        assert isinstance(self._run_mocked(), float)

    def test_returns_correct_test_metric(self):
        result = self._run_mocked(test_val=0.82)
        assert result == pytest.approx(0.82)

    def test_missing_metric_key_returns_zero(self):
        cfg = _make_cfg()
        info = _make_info("acc")
        trainer_mock = MagicMock()
        trainer_mock.test.return_value = [{}]  # no test_acc key

        with (
            patch("exp.run.SheafDataModule"),
            patch("exp.run.SheafLightningModule"),
            patch("exp.run.Trainer", return_value=trainer_mock),
            patch("exp.run.EarlyStopping"),
            patch("exp.run.ModelCheckpoint"),
            patch("exp.run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False
            result = _run_fold(cfg, info, fold=0, monitor="val_acc", ckpt_mode="max")

        assert result == pytest.approx(0.0)

    def test_roc_auc_metric_key(self):
        result = self._run_mocked(metric="roc_auc", test_val=0.91)
        assert result == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# run() function
# ---------------------------------------------------------------------------


class TestRunFunction:
    """Test the extracted run() CV loop with mocked trainer and data module."""

    def _run_mocked(
        self,
        cfg: Config | None = None,
        *,
        n_folds: int = 3,
        test_val: float = 0.75,
        metric: str = "acc",
    ) -> list[float]:
        cfg = cfg or _make_cfg()
        dm_mock = _make_dm_mock(metric=metric, num_splits=n_folds)
        trainer_mock = _make_trainer_mock(metric, test_val)

        with (
            patch("exp.run.SheafDataModule", return_value=dm_mock),
            patch("exp.run.SheafLightningModule"),
            patch("exp.run.Trainer", return_value=trainer_mock),
            patch("exp.run.EarlyStopping"),
            patch("exp.run.ModelCheckpoint"),
            patch("exp.run.setup_torch"),
            patch("exp.run._silence_third_party"),
            patch("exp.run._display_startup"),
            patch("exp.run._display_results"),
            patch("exp.run.tempfile.TemporaryDirectory") as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/ckpt"
            mock_tmpdir.return_value.__exit__.return_value = False
            return run(cfg)

    def test_returns_list_of_floats(self):
        results = self._run_mocked(n_folds=3)
        assert isinstance(results, list)
        assert all(isinstance(r, float) for r in results)

    def test_result_length_equals_n_folds(self):
        results = self._run_mocked(n_folds=4)
        assert len(results) == 4

    def test_aggregation_mean_is_correct(self):
        results = self._run_mocked(n_folds=3, test_val=0.80)
        assert np.mean(results) == pytest.approx(0.80)

    def test_min_acc_guard_aborts_after_fold_0(self):
        from exp.config import CVConfig

        cfg = _make_cfg(cv=CVConfig(folds=10, seed=42, min_acc=0.99))
        # test_val=0.50 < min_acc=0.99 -> should abort after fold 0
        results = self._run_mocked(cfg=cfg, n_folds=10, test_val=0.50)
        assert len(results) == 1

    def test_min_acc_guard_does_not_abort_when_threshold_met(self):
        from exp.config import CVConfig

        cfg = _make_cfg(cv=CVConfig(folds=3, seed=42, min_acc=0.1))
        results = self._run_mocked(cfg=cfg, n_folds=3, test_val=0.75)
        assert len(results) == 3

    def test_min_acc_guard_only_applies_to_acc_metric(self):
        from exp.config import CVConfig

        cfg = _make_cfg(cv=CVConfig(folds=3, seed=42, min_acc=0.99))
        # roc_auc metric: guard should not fire even if value < min_acc
        results = self._run_mocked(cfg=cfg, n_folds=3, test_val=0.10, metric="roc_auc")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# splits() / download_canonical_splits / generate_splits
# ---------------------------------------------------------------------------


class TestDownloadCanonicalSplits:
    """Test download_canonical_splits with mocked urllib."""

    def test_requests_correct_url_for_each_fold(self, tmp_path):
        with patch("exp.gen_splits.urllib.request.urlretrieve") as mock_retrieve:
            # Make the retrieve call write a valid npz so np.load succeeds.
            def fake_retrieve(url, out_path):
                n = 10
                np.savez(
                    out_path,
                    train_mask=np.ones(n, dtype=bool),
                    val_mask=np.zeros(n, dtype=bool),
                    test_mask=np.zeros(n, dtype=bool),
                )

            mock_retrieve.side_effect = fake_retrieve
            download_canonical_splits("cora", splits_dir=str(tmp_path), n_folds=3)

        urls = [c.args[0] for c in mock_retrieve.call_args_list]
        assert len(urls) == 3
        for fold, url in enumerate(urls):
            assert f"cora_split_0.6_0.2_{fold}.npz" in url

    def test_skips_existing_files_when_overwrite_false(self, tmp_path):
        fold_path = tmp_path / "cora_split_0.6_0.2_0.npz"
        n = 10
        np.savez(
            str(fold_path),
            train_mask=np.ones(n, dtype=bool),
            val_mask=np.zeros(n, dtype=bool),
            test_mask=np.zeros(n, dtype=bool),
        )

        with patch("exp.gen_splits.urllib.request.urlretrieve") as mock_retrieve:
            download_canonical_splits(
                "cora", splits_dir=str(tmp_path), n_folds=1, overwrite=False
            )

        mock_retrieve.assert_not_called()

    def test_overwrites_existing_files_when_overwrite_true(self, tmp_path):
        fold_path = tmp_path / "cora_split_0.6_0.2_0.npz"
        n = 10
        np.savez(
            str(fold_path),
            train_mask=np.ones(n, dtype=bool),
            val_mask=np.zeros(n, dtype=bool),
            test_mask=np.zeros(n, dtype=bool),
        )

        with patch("exp.gen_splits.urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.side_effect = lambda url, p: np.savez(
                p,
                train_mask=np.ones(n, dtype=bool),
                val_mask=np.zeros(n, dtype=bool),
                test_mask=np.zeros(n, dtype=bool),
            )
            download_canonical_splits(
                "cora", splits_dir=str(tmp_path), n_folds=1, overwrite=True
            )

        mock_retrieve.assert_called_once()


class TestGenerateSplits:
    """Test generate_splits against a small synthetic in-memory dataset."""

    def _make_fake_data(self, n: int = 30):
        import torch
        from torch_geometric.data import Data

        torch.manual_seed(0)
        return (
            Data(
                x=torch.randn(n, 5),
                edge_index=torch.randint(0, n, (2, n)),
                y=torch.randint(0, 3, (n,)),
            ),
            None,  # info not needed by generate_splits
        )

    def test_output_files_are_created(self, tmp_path):
        n = 30
        fake_data, _ = self._make_fake_data(n)

        with patch(
            "exp.gen_splits.load_dataset", return_value=(fake_data, MagicMock())
        ):
            generate_splits("toy", splits_dir=str(tmp_path), n_folds=2)

        for fold in range(2):
            assert (tmp_path / f"toy_split_0.6_0.2_{fold}.npz").exists()

    def test_masks_sum_to_n(self, tmp_path):
        n = 30
        fake_data, _ = self._make_fake_data(n)

        with patch(
            "exp.gen_splits.load_dataset", return_value=(fake_data, MagicMock())
        ):
            generate_splits("toy", splits_dir=str(tmp_path), n_folds=1)

        arr = np.load(str(tmp_path / "toy_split_0.6_0.2_0.npz"))
        total = arr["train_mask"].sum() + arr["val_mask"].sum() + arr["test_mask"].sum()
        assert total == n

    def test_masks_are_mutually_exclusive(self, tmp_path):
        n = 30
        fake_data, _ = self._make_fake_data(n)

        with patch(
            "exp.gen_splits.load_dataset", return_value=(fake_data, MagicMock())
        ):
            generate_splits("toy", splits_dir=str(tmp_path), n_folds=1)

        arr = np.load(str(tmp_path / "toy_split_0.6_0.2_0.npz"))
        assert not (arr["train_mask"] & arr["val_mask"]).any()
        assert not (arr["train_mask"] & arr["test_mask"]).any()
        assert not (arr["val_mask"] & arr["test_mask"]).any()

    def test_overwrite_flag_recreates_existing_file(self, tmp_path):
        n = 30
        fake_data, _ = self._make_fake_data(n)

        existing = tmp_path / "toy_split_0.6_0.2_0.npz"
        existing.write_bytes(b"old content")

        with patch(
            "exp.gen_splits.load_dataset", return_value=(fake_data, MagicMock())
        ):
            generate_splits("toy", splits_dir=str(tmp_path), n_folds=1, overwrite=True)

        arr = np.load(str(existing))
        assert "train_mask" in arr  # valid npz now

    def test_skips_existing_file_when_overwrite_false(self, tmp_path):
        """Lines 162-163: existing file is NOT overwritten when overwrite=False."""
        n = 30
        fake_data, _ = self._make_fake_data(n)

        existing = tmp_path / "toy_split_0.6_0.2_0.npz"
        sentinel = b"do not overwrite"
        existing.write_bytes(sentinel)

        with patch(
            "exp.gen_splits.load_dataset", return_value=(fake_data, MagicMock())
        ):
            generate_splits("toy", splits_dir=str(tmp_path), n_folds=1, overwrite=False)

        # File content must be unchanged because the fold was skipped
        assert existing.read_bytes() == sentinel, (
            "File should not be overwritten when overwrite=False"
        )


class TestSplitsFunction:
    """Test the splits() dispatcher."""

    def test_skips_non_npz_datasets(self):
        cfg = SplitsConfig(datasets=["not_a_real_dataset"], folds=1)
        # Should not raise; just skip with a warning
        with patch("exp.gen_splits._npz_split_datasets", return_value=frozenset()):
            splits(cfg)  # no error

    def test_canonical_source_calls_download(self):
        cfg = SplitsConfig(datasets=["cora"], source="canonical", folds=2)
        with (
            patch(
                "exp.gen_splits._npz_split_datasets", return_value=frozenset(["cora"])
            ),
            patch("exp.gen_splits.download_canonical_splits") as mock_dl,
        ):
            splits(cfg)
        mock_dl.assert_called_once()
        assert mock_dl.call_args.kwargs["n_folds"] == 2

    def test_generate_source_calls_generate(self):
        cfg = SplitsConfig(datasets=["cora"], source="generate", folds=2)
        with (
            patch(
                "exp.gen_splits._npz_split_datasets", return_value=frozenset(["cora"])
            ),
            patch("exp.gen_splits.generate_splits") as mock_gen,
        ):
            splits(cfg)
        mock_gen.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_config (run.py)
# ---------------------------------------------------------------------------


class TestRunParseConfig:
    """Test _parse_config() preset-stripping logic in exp.run."""

    def test_no_preset_calls_tyro_with_none_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        cfg = _make_cfg()
        with patch("exp.run.tyro.cli", return_value=cfg) as mock_cli:
            from exp.run import _parse_config

            result = _parse_config()
        assert result is cfg
        mock_cli.assert_called_once()
        _, kwargs = mock_cli.call_args
        assert kwargs["default"] is None

    def test_preset_space_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "cora"])
        from exp.run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.run.preset_registry") as mock_reg,
            patch("exp.run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg
        # argv should have --preset stripped
        assert "--preset" not in sys.argv
        assert "cora" not in sys.argv

    def test_preset_equals_syntax_loads_preset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset=cora"])
        from exp.run import _parse_config

        expected_cfg = _make_cfg()
        with (
            patch("exp.run.preset_registry") as mock_reg,
            patch("exp.run.tyro.cli", return_value=expected_cfg),
        ):
            mock_reg.__contains__ = lambda self, x: True
            mock_reg.get = lambda x: expected_cfg
            result = _parse_config()
        assert result is expected_cfg

    def test_unknown_preset_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--preset", "nonexistent_xyz"])
        from exp.run import _parse_config

        with pytest.raises(SystemExit):
            _parse_config()

    def test_preset_flag_without_value_is_skipped(self, monkeypatch):
        # --preset at end of argv with no following value — should not crash
        monkeypatch.setattr(sys, "argv", ["prog", "--preset"])
        cfg = _make_cfg()
        with patch("exp.run.tyro.cli", return_value=cfg):
            from exp.run import _parse_config

            result = _parse_config()
        assert result is cfg

    def test_non_preset_args_are_preserved(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--dataset.name", "texas"])
        cfg = _make_cfg()
        with patch("exp.run.tyro.cli", return_value=cfg):
            from exp.run import _parse_config

            _parse_config()
        assert "--dataset.name" in sys.argv
        assert "texas" in sys.argv


# ---------------------------------------------------------------------------
# _make_logger (run.py)
# ---------------------------------------------------------------------------


def test_make_logger_wandb_disabled_returns_false():
    from exp.run import _make_logger

    cfg = _make_cfg()  # wandb.enabled defaults to False
    info = _make_info()
    result = _make_logger(cfg, info, fold=0, run_name="test-run")
    assert result is False


def test_make_logger_wandb_enabled_returns_logger_instance():
    from exp.config import WandBConfig
    from exp.run import _make_logger

    cfg = _make_cfg()
    cfg.wandb = WandBConfig(enabled=True, project="my-project")
    info = _make_info()

    mock_logger_instance = MagicMock()
    mock_wandb_logger_cls = MagicMock(return_value=mock_logger_instance)

    with patch("lightning.pytorch.loggers.WandbLogger", mock_wandb_logger_cls):
        result = _make_logger(cfg, info, fold=0, run_name="test-run")

    assert result is mock_logger_instance


# ---------------------------------------------------------------------------
# _display_startup (run.py)
# ---------------------------------------------------------------------------


def test_display_startup_does_not_raise():
    from exp.run import _display_startup

    cfg = _make_cfg()
    info = _make_info()
    with (
        patch("exp.run.SheafLightningModule"),
        patch(
            "exp.run.ModelSummary", return_value=MagicMock(__str__=lambda s: "summary")
        ),
    ):
        _display_startup(
            cfg, info, n_train=100, n_val=50, n_test=50, avg_deg=2.0, n_folds=3
        )


# ---------------------------------------------------------------------------
# _display_results (run.py)
# ---------------------------------------------------------------------------


def test_display_results_acc_metric():
    from exp.run import _display_results

    cfg = _make_cfg()
    info = _make_info(metric="acc")
    _display_results(cfg, info, [0.75, 0.80, 0.82])


def test_display_results_roc_auc_metric():
    from exp.run import _display_results

    cfg = _make_cfg()
    info = _make_info(metric="roc_auc")
    _display_results(cfg, info, [0.91, 0.89])


# ---------------------------------------------------------------------------
# _silence_third_party (run.py)
# ---------------------------------------------------------------------------


def test_silence_third_party_does_not_raise():
    from exp.run import _silence_third_party

    _silence_third_party()  # should complete without error


# ---------------------------------------------------------------------------
# main() (run.py)
# ---------------------------------------------------------------------------


def test_main_calls_parse_config_and_run():
    from exp.run import main

    cfg = _make_cfg()
    with (
        patch("exp.run._parse_config", return_value=cfg) as mock_parse,
        patch("exp.run.run", return_value=[0.75]) as mock_run,
        patch("exp.run.logging.basicConfig"),
    ):
        main()

    mock_parse.assert_called_once()
    mock_run.assert_called_once_with(cfg)
