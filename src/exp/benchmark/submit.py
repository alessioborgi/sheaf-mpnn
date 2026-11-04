# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Submits the pending job matrix to Slurm via submitit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import submitit

from exp.benchmark.adapters import build_command, hparams_path
from exp.benchmark.hparams import (
    load_original_hparams,
    load_ours_hparams,
    validate_matched_architecture,
)
from exp.benchmark.jobspec import JobSpec, pending_jobs, result_path
from exp.benchmark.models import BenchmarkManifest


def run_one_job(manifest: BenchmarkManifest, job: JobSpec) -> None:
    """Executed on the Slurm worker: build and run this job's subprocess."""
    out_path = result_path(manifest.results_dir, job)
    argv, cwd = build_command(manifest, job, out_path)
    subprocess.run(argv, cwd=cwd, check=True)


def check_hparam_files_exist(manifest: BenchmarkManifest) -> None:
    """Fail fast, with every missing file listed, if any (dataset, variant)
    pair lacks a hyperparameter config on either side.

    Without this a typo in ``config_dir`` surfaces as a bare
    ``FileNotFoundError`` from deep inside architecture validation -- or, for
    a pair that isn't checked first, only once the Slurm job is already
    running.
    """
    missing: list[str] = []
    for dataset in manifest.datasets:
        for variant in manifest.variants:
            for side in ("ours", "original"):
                job = JobSpec(dataset, variant, side, "convergence", 0)
                path = hparams_path(manifest, job)
                if not path.is_file():
                    missing.append(f"  {dataset}/{variant} ({side}): {path}")
    if missing:
        raise ValueError(
            "Missing hyperparameter config file(s) for "
            f"{len(missing)} (dataset, variant, side) combination(s):\n"
            + "\n".join(missing)
        )


def _check_architectures(manifest: BenchmarkManifest, jobs: list[JobSpec]) -> None:
    """Raise before submitting anything if any pending (dataset, variant)
    pair's resolved hyperparameters don't share the same architecture size.
    """
    for dataset, variant in sorted({(j.dataset, j.variant) for j in jobs}):
        probe = JobSpec(dataset, variant, "ours", "convergence", 0)
        ours = load_ours_hparams(hparams_path(manifest, probe))
        original = load_original_hparams(
            hparams_path(
                manifest, JobSpec(dataset, variant, "original", "convergence", 0)
            )
        )
        validate_matched_architecture(ours, original, dataset=dataset, variant=variant)


def _check_slurm_placeholders(manifest: BenchmarkManifest) -> None:
    """The example manifest ships with slurm.partition/account set to the
    literal string "REQUIRED". Only submit() may reject it -- collect/report
    must keep working with a placeholder manifest -- and rejecting it here
    beats a cryptic sbatch failure after the job matrix is already built.
    """
    placeholders = [
        field
        for field in ("partition", "account")
        if getattr(manifest.slurm, field) == "REQUIRED"
    ]
    if placeholders:
        raise ValueError(
            f"manifest slurm.{' and slurm.'.join(placeholders)} still set to "
            "the 'REQUIRED' placeholder; fill in your cluster's values before "
            "submitting"
        )


def submit(manifest: BenchmarkManifest, modes: list[str] | None = None) -> list:
    """Submit every job in *manifest* that doesn't already have a result,
    optionally restricted to a subset of *modes*.
    """
    _check_slurm_placeholders(manifest)
    check_hparam_files_exist(manifest)

    jobs = pending_jobs(manifest)
    if modes is not None:
        jobs = [j for j in jobs if j.mode in modes]
    if not jobs:
        return []

    _check_architectures(manifest, jobs)

    executor = submitit.AutoExecutor(folder=str(Path(manifest.results_dir) / "logs"))
    executor.update_parameters(
        slurm_partition=manifest.slurm.partition,
        slurm_account=manifest.slurm.account,
        slurm_gres=manifest.slurm.gres,
        cpus_per_task=manifest.slurm.cpus_per_task,
        mem_gb=manifest.slurm.mem_gb,
        timeout_min=manifest.slurm.timeout_min,
        slurm_array_parallelism=manifest.slurm.slurm_array_parallelism,
    )
    return executor.map_array(lambda job: run_one_job(manifest, job), jobs)
