# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Renders the paper-ready comparison figure and results table from
``collect``'s ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp.benchmark.models import BenchmarkManifest  # noqa: E402


def _load_summary(manifest: BenchmarkManifest) -> dict:
    return json.loads((Path(manifest.results_dir) / "summary.json").read_text())


def render_figure(
    manifest: BenchmarkManifest, metric: str = "fit_time_s", mode: str = "convergence"
) -> Path:
    summary = _load_summary(manifest)
    rows = [
        s for s in summary["summaries"] if s["mode"] == mode and metric in s["means"]
    ]
    datasets = sorted({r["dataset"] for r in rows})
    variants = sorted({r["variant"] for r in rows})

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.2), 4))
    n_bars = max(len(variants) * 2, 1)
    width = 0.8 / n_bars
    for v_idx, variant in enumerate(variants):
        for s_idx, side in enumerate(("ours", "original")):
            xs, ys, yerr = [], [], []
            for d_idx, dataset in enumerate(datasets):
                match = next(
                    (
                        r
                        for r in rows
                        if (
                            r["dataset"] == dataset
                            and r["variant"] == variant
                            and r["side"] == side
                        )
                    ),
                    None,
                )
                if match is None:
                    continue
                # Center the group of bars on the dataset's tick.
                offset = (v_idx * 2 + s_idx - (n_bars - 1) / 2) * width
                xs.append(d_idx + offset)
                mean = match["means"][metric]
                std = match["stds"].get(metric, 0.0)
                ys.append(mean)
                # The y-axis is log-scaled below; clamp the lower error bar so
                # mean - err stays positive instead of being clipped noisily.
                yerr.append([min(std, mean * 0.999), std])
            ax.bar(
                xs,
                ys,
                width=width,
                yerr=list(zip(*yerr, strict=True)) if yerr else None,
                label=f"{variant}-{side}",
                capsize=2,
            )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    out_path = Path(manifest.results_dir) / f"{metric}_{mode}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_table(manifest: BenchmarkManifest) -> Path:
    summary = _load_summary(manifest)
    alpha = manifest.stats.alpha
    lines = [
        "| Dataset | Variant | Mode | Metric | p-value | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for c in summary["comparisons"]:
        if c["p_value"] is None:  # param_count: ratio only, no paired test
            p_cell = "n/a"
        elif c["p_value"] < alpha:
            p_cell = f"**{c['p_value']:.4g}**"
        else:
            p_cell = f"{c['p_value']:.4g}"
        lines.append(
            f"| {c['dataset']} | {c['variant']} | {c['mode']} | {c['metric']} | "
            f"{p_cell} | {c['speedup']:.2f}x |"
        )
    lines.append("")
    lines.append(f"Bold p-values are significant at alpha = {alpha} (Wilcoxon).")
    out_path = Path(manifest.results_dir) / "comparison_table.md"
    out_path.write_text("\n".join(lines))
    return out_path
