from __future__ import annotations

from pathlib import Path

from exp.benchmark.jobspec import expand_job_matrix, pending_jobs, result_path
from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.result import ResultRecord


def _manifest(tmp_path: Path, **overrides) -> BenchmarkManifest:
    ours_dir = tmp_path / "ours"
    original_dir = tmp_path / "original"
    ours_dir.mkdir(exist_ok=True)
    original_dir.mkdir(exist_ok=True)
    data = {
        "name": "test",
        "datasets": ["texas", "cora"],
        "variants": ["orthogonal"],
        "repeats": {"folds": 10, "efficiency_seeds": 20},
        "sides": {
            "ours": {"config_dir": str(ours_dir)},
            "original": {
                "python": "x",
                "repo_path": "external/neural-sheaf-diffusion",
                "config_dir": str(original_dir),
            },
        },
        "results_dir": str(tmp_path / "results"),
        "slurm": {"partition": "gpu", "account": "acct"},
    }
    data.update(overrides)
    return BenchmarkManifest.model_validate(data)


def test_expand_job_matrix_size(tmp_path):
    manifest = _manifest(tmp_path)
    jobs = expand_job_matrix(manifest)
    # 2 datasets * 1 variant * 2 sides * (10 folds + 20 efficiency seeds)
    assert len(jobs) == 2 * 1 * 2 * 30


def test_expand_job_matrix_respects_modes(tmp_path):
    manifest = _manifest(tmp_path, modes=["efficiency"])
    jobs = expand_job_matrix(manifest)
    assert jobs and all(job.mode == "efficiency" for job in jobs)
    assert len(jobs) == 2 * 1 * 2 * manifest.repeats.efficiency_seeds


def _write_valid_result(path: Path, job) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_pending_jobs_skips_existing_valid_results(tmp_path):
    manifest = _manifest(
        tmp_path,
        datasets=["texas"],
        modes=["efficiency"],
        repeats={"efficiency_seeds": 2},
    )
    all_jobs = expand_job_matrix(manifest)
    done = all_jobs[0]
    _write_valid_result(result_path(manifest.results_dir, done), done)

    remaining = pending_jobs(manifest)
    assert done not in remaining
    assert len(remaining) == len(all_jobs) - 1


def test_pending_jobs_resubmits_corrupt_results(tmp_path):
    """A truncated/unparseable result file must not permanently block
    resubmission -- collect.load_results already treats it as missing for
    aggregation, and submit's pending_jobs must agree so it actually
    regenerates the file, instead of silently skipping the job forever.
    """
    manifest = _manifest(
        tmp_path,
        datasets=["texas"],
        modes=["efficiency"],
        repeats={"efficiency_seeds": 2},
    )
    all_jobs = expand_job_matrix(manifest)
    corrupt = all_jobs[0]
    path = result_path(manifest.results_dir, corrupt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")  # missing required fields -> fails validation

    remaining = pending_jobs(manifest)
    assert corrupt in remaining
    assert len(remaining) == len(all_jobs)
