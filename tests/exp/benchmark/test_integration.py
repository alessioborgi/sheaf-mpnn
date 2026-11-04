# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Real, end-to-end integration tests for the efficiency benchmark: tiny
(CPU, few-epoch) training runs on both sides, and the full
submit -> collect -> report pipeline via submitit's local executor backend
(no real Slurm needed). Requires external/neural-sheaf-diffusion's own venv
(scripts/benchmark/setup_original_env.sh) for the "original"-side tests --
skipped otherwise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_PYTHON = REPO_ROOT / "external/neural-sheaf-diffusion/.venv/bin/python"
ORIGINAL_ENTRYPOINT = REPO_ROOT / "src/exp/benchmark/original_entrypoint.py"

requires_original_env = pytest.mark.skipif(
    not ORIGINAL_PYTHON.exists(),
    reason=(
        "external/neural-sheaf-diffusion/.venv missing; "
        "run scripts/benchmark/setup_original_env.sh"
    ),
)


def _ours_hparams(tmp_path: Path) -> Path:
    path = tmp_path / "ours.yaml"
    path.write_text(
        yaml.dump(
            {
                "model": "nsd",
                "best_params": {
                    "variant": "diagonal",
                    "stalk_dim": 2,
                    "hidden_dim": 4,
                    "num_layers": 1,
                    "epochs": 3,
                    "early_stopping": 3,
                },
            }
        )
    )
    return path


def _original_hparams(tmp_path: Path) -> Path:
    path = tmp_path / "original.yaml"
    path.write_text(yaml.dump({"d": 2, "hidden_channels": 4, "layers": 1}))
    return path


def test_ours_entrypoint_runs_real_tiny_training(tmp_path):
    from exp.benchmark.ours_entrypoint import run_one

    record = run_one(
        dataset="texas",
        variant="diagonal",
        mode="efficiency",
        index=0,
        hparams_path=_ours_hparams(tmp_path),
        epoch_cap=3,
        warmup_epochs=1,
    )
    assert record.epochs_trained == 3
    assert record.fit_time_s > 0
    assert record.inference_latency_ms is not None


@requires_original_env
def test_original_entrypoint_runs_real_tiny_training(tmp_path):
    from exp.benchmark.result import ResultRecord

    out_path = tmp_path / "out.json"
    subprocess.run(
        [
            str(ORIGINAL_PYTHON),
            str(ORIGINAL_ENTRYPOINT),
            "--repo-path",
            str(REPO_ROOT / "external/neural-sheaf-diffusion"),
            "--dataset",
            "texas",
            "--variant",
            "DiagSheaf",
            "--variant-label",
            "diagonal",
            "--mode",
            "efficiency",
            "--index",
            "0",
            "--hparams",
            str(_original_hparams(tmp_path)),
            "--out",
            str(out_path),
            "--epoch-cap",
            "3",
            "--warmup-epochs",
            "1",
        ],
        cwd=REPO_ROOT / "external/neural-sheaf-diffusion",
        check=True,
    )
    record = ResultRecord.read(out_path)
    assert record.side == "original"
    assert record.epochs_trained == 3
    # The canonical name, not the original repo's "DiagSheaf" class name --
    # collect() pairs the two sides on it.
    assert record.variant == "diagonal"


@requires_original_env
def test_submit_collect_report_end_to_end_with_local_executor(tmp_path):
    import submitit

    from exp.benchmark.collect import collect
    from exp.benchmark.jobspec import pending_jobs
    from exp.benchmark.models import BenchmarkManifest
    from exp.benchmark.report import render_figure, render_table
    from exp.benchmark.submit import run_one_job

    ours_dir, original_dir = (
        tmp_path / "ours" / "texas",
        tmp_path / "original" / "texas",
    )
    ours_dir.mkdir(parents=True)
    original_dir.mkdir(parents=True)
    (ours_dir / "diagonal.yaml").write_text(_ours_hparams(tmp_path).read_text())
    (original_dir / "diagonal.yaml").write_text(_original_hparams(tmp_path).read_text())

    manifest = BenchmarkManifest.model_validate(
        {
            "name": "it",
            "datasets": ["texas"],
            "variants": ["diagonal"],
            "modes": ["efficiency"],
            "repeats": {
                "efficiency_seeds": 2,
                "efficiency_epoch_cap": 3,
                "warmup_epochs": 1,
            },
            "sides": {
                "ours": {"config_dir": str(tmp_path / "ours")},
                "original": {
                    "python": str(ORIGINAL_PYTHON),
                    "repo_path": str(REPO_ROOT / "external/neural-sheaf-diffusion"),
                    "config_dir": str(tmp_path / "original"),
                },
            },
            "results_dir": str(tmp_path / "results"),
            "slurm": {"partition": "unused", "account": "unused"},
        }
    )

    executor = submitit.AutoExecutor(
        folder=str(tmp_path / "results" / "logs"), cluster="local"
    )
    executor.update_parameters(timeout_min=10)
    jobs = executor.map_array(
        lambda job: run_one_job(manifest, job), pending_jobs(manifest)
    )
    for job in jobs:
        job.result()  # raises if the job failed

    collect(manifest)
    assert (tmp_path / "results" / "summary.json").exists()
    render_table(manifest)
    render_figure(manifest, metric="fit_time_s", mode="efficiency")
    assert (tmp_path / "results" / "comparison_table.md").exists()
