# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Aggregates per-job ResultRecord JSONs into summary stats and a paired
significance test between "ours" and "original".
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from scipy import stats

from exp.benchmark.jobspec import expand_job_matrix, result_path
from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.result import ResultRecord

EFFICIENCY_METRICS = (
    "fit_time_s",
    "epoch_time_s",
    "peak_gpu_mem_mb",
    "param_count",
    "inference_latency_ms",
)


@dataclass
class CellSummary:
    dataset: str
    variant: str
    side: str
    mode: str
    n: int
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class Comparison:
    dataset: str
    variant: str
    mode: str
    metric: str
    # None for param_count: it is constant per side, so a paired significance
    # test is meaningless -- only the ratio (speedup) is reported.
    p_value: float | None
    speedup: float  # mean(original) / mean(ours); > 1 means "ours" wins


def load_results(manifest: BenchmarkManifest) -> list[ResultRecord]:
    records = []
    missing = []
    for job in expand_job_matrix(manifest):
        path = result_path(manifest.results_dir, job)
        if not path.exists():
            missing.append(job)
            continue
        try:
            records.append(ResultRecord.read(path))
        except (ValidationError, OSError) as exc:
            # A job preempted mid-write (or otherwise truncated) leaves an
            # unparseable file: treat it as missing rather than crashing the
            # whole collection, so re-running submit() can regenerate it.
            print(f"warning: unreadable result {path}: {exc}")
            missing.append(job)
    if missing:
        print(f"warning: {len(missing)} result(s) missing, e.g. {missing[0]}")
    return records


def _group(
    records: list[ResultRecord],
) -> dict[tuple[str, str, str, str], list[ResultRecord]]:
    groups: dict[tuple[str, str, str, str], list[ResultRecord]] = defaultdict(list)
    for r in records:
        groups[(r.dataset, r.variant, r.side, r.mode)].append(r)
    return groups


def summarize(records: list[ResultRecord]) -> list[CellSummary]:
    summaries = []
    for (dataset, variant, side, mode), group in _group(records).items():
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for metric in (*EFFICIENCY_METRICS, "test_metric"):
            values = [
                getattr(r, metric) for r in group if getattr(r, metric) is not None
            ]
            if values:
                means[metric] = float(np.mean(values))
                stds[metric] = float(np.std(values))
        summaries.append(
            CellSummary(dataset, variant, side, mode, len(group), means, stds)
        )
    return summaries


def compare(records: list[ResultRecord]) -> list[Comparison]:
    groups = _group(records)
    comparisons = []
    for dataset, variant, side, mode in groups:
        if side != "ours":
            continue
        ours_by_index = {r.index: r for r in groups[(dataset, variant, "ours", mode)]}
        original_by_index = {
            r.index: r for r in groups.get((dataset, variant, "original", mode), [])
        }
        shared = sorted(set(ours_by_index) & set(original_by_index))
        if len(shared) < 2:
            continue
        for metric in EFFICIENCY_METRICS:
            ours_vals = [getattr(ours_by_index[i], metric) for i in shared]
            original_vals = [getattr(original_by_index[i], metric) for i in shared]
            if any(v is None for v in (*ours_vals, *original_vals)):
                continue
            if metric == "param_count":
                p_value = None
            else:
                diffs = np.array(original_vals) - np.array(ours_vals)
                p_value = (
                    1.0
                    if np.allclose(diffs, 0.0)
                    else float(stats.wilcoxon(ours_vals, original_vals)[1])
                )
            speedup = float(np.mean(original_vals) / np.mean(ours_vals))
            comparisons.append(
                Comparison(dataset, variant, mode, metric, p_value, speedup)
            )
    return comparisons


def collect(manifest: BenchmarkManifest) -> None:
    records = load_results(manifest)
    summaries = summarize(records)
    comparisons = compare(records)

    out_dir = Path(manifest.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summaries": [asdict(s) for s in summaries],
                "comparisons": [asdict(c) for c in comparisons],
            },
            indent=2,
        )
    )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "variant", "side", "mode", "n", "metric", "mean", "std"]
        )
        for s in summaries:
            for metric in s.means:
                writer.writerow(
                    [
                        s.dataset,
                        s.variant,
                        s.side,
                        s.mode,
                        s.n,
                        metric,
                        s.means[metric],
                        s.stds[metric],
                    ]
                )
