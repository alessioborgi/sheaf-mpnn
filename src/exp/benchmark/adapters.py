# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Translates a JobSpec + manifest into the subprocess command line for
either side.
"""

from __future__ import annotations

from pathlib import Path

from exp.benchmark.jobspec import JobSpec
from exp.benchmark.models import BenchmarkManifest


def hparams_path(manifest: BenchmarkManifest, job: JobSpec) -> Path:
    if job.side == "ours":
        config_dir = manifest.sides.ours.config_dir
    else:
        config_dir = manifest.sides.original.config_dir
    return Path(config_dir, job.dataset, f"{job.variant}.yaml")


def build_command(
    manifest: BenchmarkManifest, job: JobSpec, result_path: Path
) -> tuple[list[str], Path]:
    """Return ``(argv, cwd)`` to invoke this job as a subprocess."""
    repeats = manifest.repeats
    shared_tail = [
        "--dataset",
        job.dataset,
        "--mode",
        job.mode,
        "--index",
        str(job.index),
        "--hparams",
        str(hparams_path(manifest, job)),
        "--out",
        str(result_path),
        "--epoch-cap",
        str(repeats.efficiency_epoch_cap),
        "--warmup-epochs",
        str(repeats.warmup_epochs),
    ]

    if job.side == "ours":
        ours_cfg = manifest.sides.ours
        argv = [
            ours_cfg.python,
            "-m",
            "exp.benchmark.ours_entrypoint",
            *shared_tail[:2],
            "--variant",
            job.variant,
            *shared_tail[2:],
        ]
        return argv, Path.cwd()

    original_cfg = manifest.sides.original
    argv = [
        original_cfg.python,
        str(Path(__file__).parent / "original_entrypoint.py"),
        "--repo-path",
        str(original_cfg.repo_path),
        *shared_tail[:2],
        "--variant",
        original_cfg.variant_map[job.variant],
        # The original repo's class name selects the model; the canonical
        # name is what lands in the result JSON, so collect() can pair the
        # two sides on a common (dataset, variant) key.
        "--variant-label",
        job.variant,
        *shared_tail[2:],
    ]
    return argv, Path(original_cfg.repo_path)
