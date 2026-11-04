# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Expands a ``BenchmarkManifest`` into the full job matrix, and filters out
jobs that already have a result on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.result import ResultRecord

Side = Literal["ours", "original"]
Mode = Literal["convergence", "efficiency"]


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    variant: str
    side: Side
    mode: Mode
    index: int  # fold index for "convergence"; seed offset for "efficiency"


def expand_job_matrix(manifest: BenchmarkManifest) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for dataset in manifest.datasets:
        for variant in manifest.variants:
            for side in ("ours", "original"):
                if "convergence" in manifest.modes:
                    for fold in range(manifest.repeats.folds):
                        jobs.append(
                            JobSpec(dataset, variant, side, "convergence", fold)
                        )
                if "efficiency" in manifest.modes:
                    for seed_offset in range(manifest.repeats.efficiency_seeds):
                        jobs.append(
                            JobSpec(dataset, variant, side, "efficiency", seed_offset)
                        )
    return jobs


def result_path(results_dir: Path, job: JobSpec) -> Path:
    return Path(
        results_dir,
        job.dataset,
        job.variant,
        job.side,
        job.mode,
        f"{job.index}.json",
    )


def _has_valid_result(path: Path) -> bool:
    """A job counts as done only if its result file exists AND parses.

    A truncated/corrupt file (e.g. from a preempted write) must not block
    resubmission forever -- ``collect.load_results`` already treats such a
    file as missing for aggregation purposes; ``submit`` needs the same
    view so it actually regenerates it, instead of silently skipping it.
    """
    if not path.exists():
        return False
    try:
        ResultRecord.read(path)
    except (ValidationError, OSError):
        return False
    return True


def pending_jobs(manifest: BenchmarkManifest) -> list[JobSpec]:
    """Jobs from the full matrix that don't already have a valid result JSON."""
    return [
        job
        for job in expand_job_matrix(manifest)
        if not _has_valid_result(result_path(manifest.results_dir, job))
    ]
