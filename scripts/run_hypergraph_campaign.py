"""Run the eight 250-trial studies for one dataset, sequentially and resumably.

Uses the library's normal sweep entry point. Logs are per study; SQLite stores
the completed-trial history. Run one campaign per GPU, not eight concurrent jobs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from exp.datasets.hypergraph import HYPERGRAPHS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=HYPERGRAPHS, required=True)
    parser.add_argument("--model", choices=("hsnn", "dhsnn"))
    parser.add_argument(
        "--variant", choices=("diagonal", "orthogonal", "low_rank", "general")
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--splits", type=int, choices=range(1, 11))
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--tag",
        default="mp-v2",
        help="Separate study namespace; keep unchanged to resume",
    )
    args = parser.parse_args()
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive")
    if not args.tag or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.tag
    ):
        parser.error("--tag must contain only letters, digits, '-' or '_'")
    root = Path("runs/hypergraph-sweeps")
    root.mkdir(parents=True, exist_ok=True)
    for model in [args.model] if args.model else ("hsnn", "dhsnn"):
        for variant in (
            [args.variant]
            if args.variant
            else ("diagonal", "orthogonal", "low_rank", "general")
        ):
            config = (
                Path("configs/hypergraphs") / args.dataset / model / f"{variant}.yaml"
            )
            spec = yaml.safe_load(config.read_text(encoding="utf-8"))
            study = f"{args.dataset}-{model}-{variant}-{args.tag}"
            spec["config"].update(
                study_name=study,
                storage=f"sqlite:///{root.as_posix()}/{study}.db",
                cuda=args.cuda,
            )
            if args.trials is not None:
                spec["config"]["n_trials"] = args.trials
            if args.splits is not None:
                spec["config"]["n_seeds_per_trial"] = args.splits
            launch = root / "launch-configs" / f"{study}.yaml"
            launch.parent.mkdir(exist_ok=True)
            launch.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "exp.cli",
                "sweep",
                "--yaml-path",
                str(launch),
                "--preset",
                args.dataset,
            ]
            log = root / f"{study}.log"
            print(f"Running {study}; log: {log}", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True
                )


if __name__ == "__main__":
    main()
