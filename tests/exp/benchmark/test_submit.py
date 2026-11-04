from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from exp.benchmark.jobspec import expand_job_matrix, result_path
from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.result import ResultRecord
from exp.benchmark.submit import submit


def _write_hparams(
    ours_dir,
    original_dir,
    *,
    dataset="texas",
    variant="orthogonal",
    stalk_dim=4,
    d=4,
    hidden_dim=16,
    hidden_channels=16,
    num_layers=2,
    layers=2,
) -> None:
    (ours_dir / dataset).mkdir(exist_ok=True)
    (ours_dir / dataset / f"{variant}.yaml").write_text(
        yaml.dump(
            {
                "model": "nsd",
                "best_params": {
                    "variant": variant,
                    "stalk_dim": stalk_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                },
            }
        )
    )
    (original_dir / dataset).mkdir(exist_ok=True)
    (original_dir / dataset / f"{variant}.yaml").write_text(
        yaml.dump({"d": d, "hidden_channels": hidden_channels, "layers": layers})
    )


def _manifest(tmp_path, **repeats_overrides) -> BenchmarkManifest:
    ours_dir, original_dir = tmp_path / "ours", tmp_path / "original"
    ours_dir.mkdir()
    original_dir.mkdir()
    _write_hparams(ours_dir, original_dir)
    return BenchmarkManifest.model_validate(
        {
            "name": "t",
            "datasets": ["texas"],
            "variants": ["orthogonal"],
            "repeats": {"folds": 2, "efficiency_seeds": 2, **repeats_overrides},
            "sides": {
                "ours": {"config_dir": str(ours_dir)},
                "original": {
                    "python": "x",
                    "repo_path": "y",
                    "config_dir": str(original_dir),
                },
            },
            "results_dir": str(tmp_path / "results"),
            "slurm": {"partition": "gpu", "account": "acct"},
        }
    )


def test_submit_calls_map_array_with_all_pending_jobs(tmp_path):
    manifest = _manifest(tmp_path)
    executor_mock = MagicMock()
    with patch(
        "exp.benchmark.submit.submitit.AutoExecutor", return_value=executor_mock
    ):
        submit(manifest)
    executor_mock.update_parameters.assert_called_once()
    # AutoExecutor deprecates the bare "gres" kwarg; the slurm_-prefixed
    # name is the stable spelling.
    kwargs = executor_mock.update_parameters.call_args.kwargs
    assert kwargs["slurm_gres"] == "gpu:1"
    assert "gres" not in kwargs
    assert executor_mock.map_array.call_count == 1
    _fn, jobs_arg = executor_mock.map_array.call_args[0]
    # 1 dataset * 1 variant * 2 sides * (2 folds + 2 efficiency seeds)
    assert len(jobs_arg) == 1 * 1 * 2 * 4


def test_submit_filters_by_mode(tmp_path):
    manifest = _manifest(tmp_path)
    executor_mock = MagicMock()
    with patch(
        "exp.benchmark.submit.submitit.AutoExecutor", return_value=executor_mock
    ):
        submit(manifest, modes=["efficiency"])
    _fn, jobs_arg = executor_mock.map_array.call_args[0]
    assert jobs_arg and all(j.mode == "efficiency" for j in jobs_arg)


def test_submit_returns_empty_and_skips_executor_when_nothing_pending(tmp_path):
    manifest = _manifest(tmp_path)
    for job in expand_job_matrix(manifest):
        path = result_path(manifest.results_dir, job)
        ResultRecord(
            dataset=job.dataset,
            variant=job.variant,
            side=job.side,
            mode=job.mode,
            index=job.index,
            fit_time_s=1.0,
            epoch_time_s=0.1,
            epochs_trained=10,
            param_count=100,
            test_metric=0.8,
            metric_name="acc",
        ).write(path)

    with patch("exp.benchmark.submit.submitit.AutoExecutor") as executor_cls:
        result = submit(manifest)

    assert result == []
    executor_cls.assert_not_called()


def test_submit_reports_every_missing_hparams_file_at_once(tmp_path):
    manifest = _manifest(tmp_path)
    _write_hparams(
        manifest.sides.ours.config_dir,
        manifest.sides.original.config_dir,
        dataset="cornell",
    )
    # texas/orthogonal exists on both sides; break it on the "ours" side only,
    # and drop cornell/orthogonal on the "original" side. Both must be named.
    (manifest.sides.ours.config_dir / "texas" / "orthogonal.yaml").unlink()
    (manifest.sides.original.config_dir / "cornell" / "orthogonal.yaml").unlink()
    manifest.datasets = ["texas", "cornell"]

    with patch("exp.benchmark.submit.submitit.AutoExecutor") as executor_cls:
        with pytest.raises(
            ValueError, match="Missing hyperparameter config"
        ) as excinfo:
            submit(manifest)

    message = str(excinfo.value)
    assert "texas/orthogonal (ours)" in message
    assert "cornell/orthogonal (original)" in message
    executor_cls.assert_not_called()


def test_submit_rejects_required_placeholder_slurm_fields(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.slurm.partition = "REQUIRED"
    with patch("exp.benchmark.submit.submitit.AutoExecutor") as executor_cls:
        with pytest.raises(ValueError, match="REQUIRED"):
            submit(manifest)
    executor_cls.assert_not_called()


def test_submit_rejects_mismatched_architecture_before_submitting(tmp_path):
    manifest = _manifest(tmp_path)
    _write_hparams(
        manifest.sides.ours.config_dir, manifest.sides.original.config_dir, stalk_dim=8
    )
    with patch("exp.benchmark.submit.submitit.AutoExecutor") as executor_cls:
        with pytest.raises(ValueError, match="Architecture mismatch"):
            submit(manifest)
    executor_cls.assert_not_called()
