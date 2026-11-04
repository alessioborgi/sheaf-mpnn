from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from exp.benchmark.ours_entrypoint import _steady_state_epoch_time, run_one
from exp.data import DatasetInfo


class _FakeEfficiencyTrainer:
    """Stands in for Lightning's Trainer in efficiency mode: drives the
    _EpochTimer callback directly and tolerates any extra Trainer kwargs.
    """

    def __init__(self, *, max_epochs, callbacks, **_):
        self.max_epochs = max_epochs
        self.timer = callbacks[0]

    def fit(self, module, dm):
        for _ in range(self.max_epochs):
            self.timer.on_train_epoch_start(self, None)
            self.timer.on_train_epoch_end(self, None)

    def test(self, module, dm, verbose=False):
        return [{"test_acc": 0.5}]


def test_steady_state_epoch_time_discards_warmup():
    assert _steady_state_epoch_time([10.0, 5.0, 1.0, 1.0, 1.0], warmup_epochs=2) == 1.0


def test_steady_state_epoch_time_falls_back_when_all_epochs_are_warmup():
    assert _steady_state_epoch_time([5.0, 5.0], warmup_epochs=10) == 5.0


def test_steady_state_epoch_time_empty_list_returns_zero():
    assert _steady_state_epoch_time([], warmup_epochs=5) == 0.0


def _make_dm_mock(metric: str = "acc") -> MagicMock:
    dm = MagicMock()
    dm.info = DatasetInfo(
        name="texas",
        num_features=5,
        num_classes=3,
        num_splits=10,
        metric=metric,
        split_type="npz_file",
    )
    return dm


def _write_ours_hparams(tmp_path):
    import yaml

    path = tmp_path / "orthogonal.yaml"
    path.write_text(
        yaml.dump(
            {
                "model": "nsd",
                "best_params": {
                    "variant": "orthogonal",
                    "stalk_dim": 4,
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "epochs": 5,
                    "early_stopping": 2,
                },
            }
        )
    )
    return path


def test_run_one_rejects_variant_mismatched_with_hparams_file(tmp_path):
    hparams_path = _write_ours_hparams(tmp_path)  # best_params say "orthogonal"
    with pytest.raises(ValueError, match="does not match best_params variant"):
        run_one(
            dataset="texas",
            variant="diagonal",
            mode="efficiency",
            index=0,
            hparams_path=hparams_path,
        )


def test_run_one_convergence_mode_uses_early_stopping_epoch_count(tmp_path):
    hparams_path = _write_ours_hparams(tmp_path)
    dm_mock = _make_dm_mock()
    trainer_mock = MagicMock()
    trainer_mock.current_epoch = 7
    trainer_mock.test.return_value = [{"test_acc": 0.83}]

    with (
        patch("exp.benchmark.ours_entrypoint.SheafDataModule", return_value=dm_mock),
        patch("exp.benchmark.ours_entrypoint.SheafLightningModule") as module_cls,
        patch("exp.benchmark.ours_entrypoint.Trainer", return_value=trainer_mock),
        patch("exp.benchmark.ours_entrypoint.EarlyStopping"),
        patch("exp.benchmark.ours_entrypoint.ModelCheckpoint"),
    ):
        module_cls.return_value.model.parameters.return_value = [
            MagicMock(numel=lambda: 10)
        ]
        record = run_one(
            dataset="texas",
            variant="orthogonal",
            mode="convergence",
            index=0,
            hparams_path=hparams_path,
        )

    assert record.side == "ours"
    assert record.mode == "convergence"
    assert record.epochs_trained == 7
    assert record.test_metric == 0.83
    assert record.inference_latency_ms is None


def test_run_one_efficiency_mode_runs_exactly_epoch_cap(tmp_path):
    hparams_path = _write_ours_hparams(tmp_path)
    dm_mock = _make_dm_mock()
    fake_batch = MagicMock()
    dm_mock.test_dataloader.return_value = [fake_batch]

    with (
        patch("exp.benchmark.ours_entrypoint.SheafDataModule", return_value=dm_mock),
        patch("exp.benchmark.ours_entrypoint.SheafLightningModule") as module_cls,
        patch("exp.benchmark.ours_entrypoint.Trainer", _FakeEfficiencyTrainer),
    ):
        module_cls.return_value.model.parameters.return_value = [
            MagicMock(numel=lambda: 10)
        ]
        record = run_one(
            dataset="texas",
            variant="orthogonal",
            mode="efficiency",
            index=0,
            hparams_path=hparams_path,
            epoch_cap=6,
            warmup_epochs=2,
        )

    assert record.mode == "efficiency"
    assert record.epochs_trained == 6
    assert record.inference_latency_ms is not None


def test_run_one_efficiency_mode_moves_module_and_batch_to_device(tmp_path):
    # Regression: Lightning's teardown returns the module to the CPU after
    # fit/test and the test dataloader yields CPU batches, so without an
    # explicit .to(device) the inference-latency timing silently measured
    # the CPU even on a GPU node.
    hparams_path = _write_ours_hparams(tmp_path)
    dm_mock = _make_dm_mock()
    fake_batch = MagicMock()
    dm_mock.test_dataloader.return_value = [fake_batch]

    with (
        patch("exp.benchmark.ours_entrypoint.SheafDataModule", return_value=dm_mock),
        patch("exp.benchmark.ours_entrypoint.SheafLightningModule") as module_cls,
        patch("exp.benchmark.ours_entrypoint.Trainer", _FakeEfficiencyTrainer),
    ):
        module = module_cls.return_value
        module.model.parameters.return_value = [MagicMock(numel=lambda: 10)]
        run_one(
            dataset="texas",
            variant="orthogonal",
            mode="efficiency",
            index=0,
            hparams_path=hparams_path,
            epoch_cap=2,
            warmup_epochs=1,
        )

    expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module.to.assert_called_once_with(expected)
    fake_batch.to.assert_called_once_with(expected)
    # ... and the timed forward pass ran on the moved batch, not the CPU one.
    module._forward.assert_called_once_with(fake_batch.to.return_value)


def test_run_one_efficiency_mode_disables_validation(tmp_path):
    # Symmetry with the original side, which times train_epoch() only:
    # Lightning runs the val loop between on_train_epoch_start/end, so it
    # must be disabled or per-epoch times include validation cost.
    hparams_path = _write_ours_hparams(tmp_path)
    dm_mock = _make_dm_mock()
    dm_mock.test_dataloader.return_value = [MagicMock()]
    seen: dict = {}

    class _RecordingTrainer(_FakeEfficiencyTrainer):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    with (
        patch("exp.benchmark.ours_entrypoint.SheafDataModule", return_value=dm_mock),
        patch("exp.benchmark.ours_entrypoint.SheafLightningModule") as module_cls,
        patch("exp.benchmark.ours_entrypoint.Trainer", _RecordingTrainer),
    ):
        module_cls.return_value.model.parameters.return_value = [
            MagicMock(numel=lambda: 10)
        ]
        run_one(
            dataset="texas",
            variant="orthogonal",
            mode="efficiency",
            index=0,
            hparams_path=hparams_path,
            epoch_cap=2,
            warmup_epochs=1,
        )

    assert seen["limit_val_batches"] == 0
    assert seen["num_sanity_val_steps"] == 0


def test_run_one_efficiency_mode_fit_time_excludes_test_call(tmp_path):
    # Convergence mode snapshots resource_metrics before trainer.test(), so
    # efficiency mode must too or fit_time_s isn't comparable across modes.
    hparams_path = _write_ours_hparams(tmp_path)
    dm_mock = _make_dm_mock()
    dm_mock.test_dataloader.return_value = [MagicMock()]
    calls: list[str] = []

    class _OrderRecordingTrainer(_FakeEfficiencyTrainer):
        def test(self, module, dm, verbose=False):
            calls.append("test")
            return super().test(module, dm, verbose=verbose)

    def _spy_resource_metrics(start, device, epochs):
        calls.append("resource_metrics")
        return {"fit_time_s": 1.0, "epoch_time_s": 0.1, "epochs_trained": float(epochs)}

    with (
        patch("exp.benchmark.ours_entrypoint.SheafDataModule", return_value=dm_mock),
        patch("exp.benchmark.ours_entrypoint.SheafLightningModule") as module_cls,
        patch("exp.benchmark.ours_entrypoint.Trainer", _OrderRecordingTrainer),
        patch(
            "exp.benchmark.ours_entrypoint.resource_metrics",
            side_effect=_spy_resource_metrics,
        ),
    ):
        module_cls.return_value.model.parameters.return_value = [
            MagicMock(numel=lambda: 10)
        ]
        run_one(
            dataset="texas",
            variant="orthogonal",
            mode="efficiency",
            index=0,
            hparams_path=hparams_path,
            epoch_cap=2,
            warmup_epochs=1,
        )

    assert calls == ["resource_metrics", "test"]
