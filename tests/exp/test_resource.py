from __future__ import annotations

import pytest

from exp.resource import resource_metrics, start_resource_tracking


def test_start_resource_tracking_returns_a_float():
    assert isinstance(start_resource_tracking(device=0), float)


def test_resource_metrics_reports_epoch_time_and_epochs_trained():
    start = start_resource_tracking(device=0)
    metrics = resource_metrics(start, device=0, epochs=10)
    assert metrics["epochs_trained"] == 10.0
    assert metrics["epoch_time_s"] == pytest.approx(metrics["fit_time_s"] / 10)


def test_resource_metrics_guards_against_zero_epochs():
    start = start_resource_tracking(device=0)
    metrics = resource_metrics(start, device=0, epochs=0)
    assert metrics["epoch_time_s"] == pytest.approx(metrics["fit_time_s"])


def test_resource_metrics_omits_gpu_memory_without_cuda(monkeypatch):
    monkeypatch.setattr("exp.resource.torch.cuda.is_available", lambda: False)
    start = start_resource_tracking(device=0)
    metrics = resource_metrics(start, device=0, epochs=5)
    assert "peak_gpu_mem_mb" not in metrics
