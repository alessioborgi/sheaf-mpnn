# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

"""Unified ``sheaf`` CLI — thin dispatch layer over exp.run, exp.gen_splits,
and exp.sweeps.sweep.

Usage
-----
    sheaf run     [--preset <name>] [config overrides...]
    sheaf splits  [splits config args...]
    sheaf sweep   --yaml-path <path> [--preset <name>]
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


def main() -> None:
    """Entry point for the ``sheaf`` shell command."""
    argv = list(sys.argv[1:])

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    subcmd, *rest = argv

    if subcmd not in ("run", "splits", "sweep"):
        _print_help()
        raise SystemExit(
            f"\nUnknown subcommand {subcmd!r}. Choose from: run, splits, sweep"
        )

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


def _print_help() -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            "[bold]sheaf run[/bold]     Run 10-fold cross-validation\n"
            "[bold]sheaf splits[/bold]  Download or generate dataset splits\n"
            "[bold]sheaf sweep[/bold]   YAML-driven hyperparameter sweep\n\n"
            "Add [cyan]--help[/cyan] after a subcommand for per-command options.",
            title="sheaf — Sheaf Neural Networks CLI",
        )
    )


if __name__ == "__main__":
    main()
