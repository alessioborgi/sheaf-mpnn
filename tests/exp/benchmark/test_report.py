from __future__ import annotations

import json

from exp.benchmark.models import BenchmarkManifest
from exp.benchmark.report import render_figure, render_table


def _manifest_with_summary(tmp_path):
    ours_dir, original_dir = tmp_path / "ours", tmp_path / "original"
    ours_dir.mkdir()
    original_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    manifest = BenchmarkManifest.model_validate(
        {
            "name": "t",
            "datasets": ["texas"],
            "variants": ["orthogonal"],
            "sides": {
                "ours": {"config_dir": str(ours_dir)},
                "original": {
                    "python": "x",
                    "repo_path": "y",
                    "config_dir": str(original_dir),
                },
            },
            "results_dir": str(results_dir),
            "slurm": {"partition": "gpu", "account": "acct"},
        }
    )
    summary = {
        "summaries": [
            {
                "dataset": "texas",
                "variant": "orthogonal",
                "side": "ours",
                "mode": "convergence",
                "n": 10,
                "means": {"fit_time_s": 5.0},
                "stds": {"fit_time_s": 0.5},
            },
            {
                "dataset": "texas",
                "variant": "orthogonal",
                "side": "original",
                "mode": "convergence",
                "n": 10,
                "means": {"fit_time_s": 10.0},
                "stds": {"fit_time_s": 1.0},
            },
        ],
        "comparisons": [
            {
                "dataset": "texas",
                "variant": "orthogonal",
                "mode": "convergence",
                "metric": "fit_time_s",
                "p_value": 0.002,
                "speedup": 2.0,
            },
            {
                "dataset": "texas",
                "variant": "orthogonal",
                "mode": "convergence",
                "metric": "param_count",
                "p_value": None,
                "speedup": 1.0,
            },
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary))
    return manifest


def test_render_figure_writes_pdf(tmp_path):
    manifest = _manifest_with_summary(tmp_path)
    path = render_figure(manifest, metric="fit_time_s", mode="convergence")
    assert path.exists()
    assert path.suffix == ".pdf"


def test_render_table_includes_speedup_and_dataset(tmp_path):
    manifest = _manifest_with_summary(tmp_path)
    content = render_table(manifest).read_text()
    assert "2.00x" in content
    assert "texas" in content


def test_render_table_marks_significance_and_skips_param_count_p(tmp_path):
    manifest = _manifest_with_summary(tmp_path)
    content = render_table(manifest).read_text()
    # 0.002 < the default alpha of 0.05 -> bold; param_count has no test.
    assert "**0.002**" in content
    assert "n/a" in content
    assert "alpha = 0.05" in content
