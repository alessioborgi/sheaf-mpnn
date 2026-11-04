# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Wall-clock and peak-GPU-memory instrumentation shared by ``exp.sweeps.sweep``
and ``exp.benchmark``.
"""

from __future__ import annotations

import time

import torch


def start_resource_tracking(device: int) -> float:
    """Reset the CUDA peak-memory counter and mark the start time."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def resource_metrics(start: float, device: int, epochs: int) -> dict[str, float]:
    """Wall-clock and peak-GPU-memory cost since ``start``."""
    elapsed = time.perf_counter() - start
    metrics = {
        "fit_time_s": elapsed,
        "epoch_time_s": elapsed / max(int(epochs), 1),
        "epochs_trained": float(epochs),
    }
    if torch.cuda.is_available():
        metrics["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return metrics
