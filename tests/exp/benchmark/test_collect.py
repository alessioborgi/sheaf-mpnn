from __future__ import annotations

import math
from typing import Literal

import pytest

from exp.benchmark.adapters import build_command
from exp.benchmark.collect import compare, load_results, summarize
from exp.benchmark.jobspec import JobSpec, result_path
from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.result import ResultRecord


def _record(
    side: Literal["ours", "original"],
    index: int,
    fit_time_s: float,
    mode: Literal["convergence", "efficiency"] = "efficiency",
) -> ResultRecord:
    return ResultRecord(
        dataset="texas",
        variant="orthogonal",
        side=side,
        mode=mode,
        index=index,
        fit_time_s=fit_time_s,
        epoch_time_s=fit_time_s / 10,
        epochs_trained=10,
        param_count=100,
        test_metric=0.8,
        metric_name="acc",
    )


def test_summarize_computes_mean_and_std():
    records = [_record("ours", i, 10.0 + i) for i in range(3)]
    [summary] = summarize(records)
    assert summary.n == 3
    assert summary.means["fit_time_s"] == pytest.approx(11.0)


def test_summarize_groups_by_dataset_variant_side_and_mode():
    records = [_record("ours", 0, 5.0), _record("original", 0, 10.0)]
    summaries = summarize(records)
    assert {s.side for s in summaries} == {"ours", "original"}


def test_compare_pairs_by_index_and_computes_speedup():
    records = [_record("ours", i, 5.0) for i in range(5)] + [
        _record("original", i, 10.0) for i in range(5)
    ]
    comparisons = compare(records)
    fit_time = next(c for c in comparisons if c.metric == "fit_time_s")
    assert fit_time.speedup == pytest.approx(2.0)


def test_compare_reports_param_count_ratio_without_p_value():
    # param_count is constant per side, so a paired significance test is
    # meaningless -- only the ratio is reported.
    records = [_record("ours", i, 5.0) for i in range(3)] + [
        _record("original", i, 10.0) for i in range(3)
    ]
    param = next(c for c in compare(records) if c.metric == "param_count")
    assert param.p_value is None
    assert param.speedup == pytest.approx(1.0)


def test_compare_skips_metrics_with_missing_values():
    # `_record` never sets peak_gpu_mem_mb, so it defaults to None on every
    # fixture record -- compare() must skip a metric entirely rather than
    # error or silently treat None as 0 when pairing ours vs. original.
    ours = [_record("ours", i, 5.0) for i in range(3)]
    original = [_record("original", i, 10.0) for i in range(3)]
    comparisons = compare(ours + original)
    assert all(c.metric != "peak_gpu_mem_mb" for c in comparisons)


def _manifest(tmp_path) -> BenchmarkManifest:
    ours_dir, original_dir = tmp_path / "ours", tmp_path / "original"
    ours_dir.mkdir()
    original_dir.mkdir()
    return BenchmarkManifest.model_validate(
        {
            "name": "t",
            "datasets": ["texas"],
            "variants": ["orthogonal"],
            "modes": ["efficiency"],
            "repeats": {"efficiency_seeds": 4},
            "sides": {
                "ours": {"config_dir": str(ours_dir)},
                "original": {
                    "python": "/env/original/bin/python",
                    "repo_path": str(tmp_path / "repo"),
                    "config_dir": str(original_dir),
                },
            },
            "results_dir": str(tmp_path / "results"),
            "slurm": {"partition": "gpu", "account": "acct"},
        }
    )


def _recorded_variant(argv: list[str]) -> str:
    """The variant string the side invoked by *argv* would write into its
    result JSON, mirroring original_entrypoint.main()'s
    ``args.variant_label or args.variant``.
    """
    if "--variant-label" in argv:
        return argv[argv.index("--variant-label") + 1]
    return argv[argv.index("--variant") + 1]


def test_variant_label_from_build_command_pairs_with_ours_side(tmp_path):
    # End-to-end guard for the class of bug where the two sides silently
    # never match: take the variant each side would actually record (derived
    # from the real adapters.build_command argv) rather than assuming both
    # equal the canonical name, and confirm compare() still pairs them.
    manifest = _manifest(tmp_path)
    records: list[ResultRecord] = []
    for index in range(4):
        ours_job = JobSpec("texas", "orthogonal", "ours", "efficiency", index)
        original_job = JobSpec("texas", "orthogonal", "original", "efficiency", index)
        ours_argv, _ = build_command(
            manifest, ours_job, result_path(manifest.results_dir, ours_job)
        )
        original_argv, _ = build_command(
            manifest, original_job, result_path(manifest.results_dir, original_job)
        )
        records.append(_record("ours", index, 5.0 + index))
        records[-1].variant = _recorded_variant(ours_argv)
        records.append(_record("original", index, 10.0 + index))
        records[-1].variant = _recorded_variant(original_argv)

    comparisons = compare(records)
    fit_time = [c for c in comparisons if c.metric == "fit_time_s"]
    assert fit_time, "ours/original never paired -- variant labels disagree"
    assert math.isfinite(fit_time[0].speedup) and fit_time[0].speedup > 1
    assert fit_time[0].p_value is not None and math.isfinite(fit_time[0].p_value)


def test_load_results_skips_corrupted_json_without_raising(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    jobs = [
        JobSpec("texas", "orthogonal", side, "efficiency", i)
        for side in ("ours", "original")
        for i in range(4)
    ]
    for job in jobs:
        _record(job.side, job.index, 5.0).write(result_path(manifest.results_dir, job))

    # Simulate a preemption partway through writing one result.
    corrupted = result_path(manifest.results_dir, jobs[0])
    corrupted.write_text('{"dataset": "texas", "vari')

    records = load_results(manifest)
    assert len(records) == len(jobs) - 1
    assert "unreadable result" in capsys.readouterr().out
