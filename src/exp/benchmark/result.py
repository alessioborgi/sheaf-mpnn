# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""The per-job result schema written by both sides' entrypoints and read
back by ``exp.benchmark.collect``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ResultRecord(BaseModel):
    dataset: str
    variant: str
    side: Literal["ours", "original"]
    mode: Literal["convergence", "efficiency"]
    index: int
    fit_time_s: float
    epoch_time_s: float
    epochs_trained: int
    peak_gpu_mem_mb: float | None = None
    param_count: int
    inference_latency_ms: float | None = None
    test_metric: float
    metric_name: str

    def write(self, path: Path) -> None:
        """Write this record atomically.

        ``pending_jobs`` treats any file at *path* as a finished job, so a
        Slurm preemption partway through a plain ``write_text`` would leave a
        truncated JSON that is never retried and later fails to parse. Write
        to a sibling temp file and ``os.replace`` it into place instead, so
        the result only ever becomes visible complete.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        os.replace(tmp, path)

    @classmethod
    def read(cls, path: Path) -> ResultRecord:
        return cls.model_validate_json(Path(path).read_text())
