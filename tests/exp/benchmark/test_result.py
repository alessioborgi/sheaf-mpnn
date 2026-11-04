from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from exp.benchmark.result import ResultRecord


def _record(**overrides: Any) -> ResultRecord:
    base = dict(
        dataset="texas",
        variant="orthogonal",
        side="ours",
        mode="convergence",
        index=0,
        fit_time_s=12.3,
        epoch_time_s=0.5,
        epochs_trained=25,
        param_count=1234,
        test_metric=0.81,
        metric_name="acc",
    )
    base.update(overrides)
    return ResultRecord(**base)  # type: ignore


def test_round_trips_through_json(tmp_path):
    record = _record(peak_gpu_mem_mb=512.0)
    path = tmp_path / "nested" / "fold0.json"
    record.write(path)
    assert ResultRecord.read(path) == record


def test_write_is_atomic_via_temp_file_and_replace(tmp_path):
    # A preempted job must never leave a truncated JSON at the final path,
    # since pending_jobs() would then treat it as a finished result forever.
    record = _record()
    path = tmp_path / "fold0.json"

    observed: list[bool] = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        # Before the rename, the payload lives at a temp path and the final
        # path does not exist yet.
        observed.append(Path(dst).exists())
        assert Path(src) != Path(dst)
        assert Path(src).parent == Path(dst).parent
        real_replace(src, dst)

    with patch("exp.benchmark.result.os.replace", _spy_replace):
        record.write(path)

    assert observed == [False]
    assert ResultRecord.read(path) == record
    assert list(tmp_path.iterdir()) == [path]


def test_optional_fields_default_to_none():
    record = _record()
    assert record.peak_gpu_mem_mb is None
    assert record.inference_latency_ms is None
