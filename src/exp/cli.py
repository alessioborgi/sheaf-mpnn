# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Unified ``sheaf`` CLI - thin dispatch layer over exp.run, exp.gen_splits,
and exp.sweeps.sweep.

Usage
-----
    sheaf run       [--preset <name>] [config overrides...]
    sheaf splits    [splits config args...]
    sheaf sweep     --yaml-path <path> [--preset <name>]
    sheaf benchmark <submit|collect|report> --manifest <path> [--modes a,b]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class _SweepArgs:
    """Arguments for ``sheaf sweep``."""

    yaml_path: Path
    preset: str | None = None


@dataclass
class _BenchmarkArgs:
    """Arguments for ``sheaf benchmark <submit|collect|report>``."""

    manifest: Path
    modes: str | None = None  # comma-separated override, e.g. "efficiency"


def main() -> None:
    """Entry point for the ``sheaf`` shell command."""
    argv = list(sys.argv[1:])

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    subcmd, *rest = argv

    if subcmd not in ("run", "splits", "sweep", "benchmark"):
        _print_help()
        raise SystemExit(
            f"\nUnknown subcommand {subcmd!r}. "
            "Choose from: run, splits, sweep, benchmark"
        )

    # Update argv so each subcommand's own tyro.cli sees a clean argument list.
    sys.argv = [f"sheaf {subcmd}"] + rest

    if subcmd == "run":
        from exp.run import _parse_config, run

        cfg = _parse_config()
        run(cfg)

    elif subcmd == "splits":
        from exp.gen_splits import SplitsConfig, splits

        cfg = tyro.cli(SplitsConfig)
        splits(cfg)

    elif subcmd == "sweep":
        from exp.sweeps.sweep import sweep

        args = tyro.cli(_SweepArgs)
        sweep(yaml_path=args.yaml_path, preset=args.preset)

    elif subcmd == "benchmark":
        _dispatch_benchmark(rest)


def _dispatch_benchmark(rest: list[str]) -> None:
    if not rest or rest[0] not in ("submit", "collect", "report"):
        raise SystemExit(
            "Usage: sheaf benchmark <submit|collect|report> "
            "--manifest <path> [--modes a,b]"
        )
    action, *action_rest = rest
    sys.argv = [f"sheaf benchmark {action}"] + action_rest

    from exp.benchmark.models import load_manifest

    args = tyro.cli(_BenchmarkArgs)
    manifest = load_manifest(args.manifest)
    modes = args.modes.split(",") if args.modes else None
    if modes is not None:
        unknown = sorted(set(modes) - {"convergence", "efficiency"})
        if unknown:
            raise SystemExit(
                f"Unknown --modes value(s) {unknown}; "
                "must be a subset of ['convergence', 'efficiency']"
            )

    if action == "submit":
        from exp.benchmark.submit import submit

        submit(manifest, modes=modes)
    elif action == "collect":
        from exp.benchmark.collect import collect

        collect(manifest)
    else:
        from exp.benchmark.report import render_figure, render_table

        render_table(manifest)
        for mode in modes or manifest.modes:
            render_figure(manifest, mode=mode)
            if mode == "efficiency":
                # These metrics only carry meaning for epoch-capped runs;
                # convergence rows don't even record inference latency.
                for metric in (
                    "epoch_time_s",
                    "inference_latency_ms",
                    "peak_gpu_mem_mb",
                ):
                    render_figure(manifest, metric=metric, mode=mode)


def _print_help() -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            "[bold]sheaf run[/bold]       Run 10-fold cross-validation\n"
            "[bold]sheaf splits[/bold]    Download or generate dataset splits\n"
            "[bold]sheaf sweep[/bold]     YAML-driven hyperparameter sweep\n"
            "[bold]sheaf benchmark[/bold] Submit/collect/report benchmark results\n\n"
            "Add [cyan]--help[/cyan] after a subcommand for per-command options.",
            title="sheaf - Sheaf Neural Networks CLI",
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
