"""Validate and summarize the preregistered paired DHSNN Cora experiments."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(root: Path) -> dict:
    """Require comparable frozen artifacts and report q=.25 minus q=0."""
    summaries = [
        json.loads((root / name / "summary.json").read_text("utf-8"))
        for name in ("q0", "q025")
    ]
    configs = [
        json.loads((root / name / "config.json").read_text("utf-8"))
        for name in ("q0", "q025")
    ]
    if configs[0]["source_sha256"] != configs[1]["source_sha256"]:
        raise ValueError("experiments used different training sources")
    if configs[0]["dataset_sha256"] != configs[1]["dataset_sha256"]:
        raise ValueError("experiments used different datasets")
    if any(config.get("model") != "dhsnn" for config in configs):
        raise ValueError("both experiment models must be dhsnn")
    for key in (
        "model",
        "runs",
        "seed",
        "epochs",
        "patience",
        "variant",
        "stalk_dim",
        "hidden_dim",
        "num_layers",
        "input_dropout",
        "rank",
        "orth_strategy",
        "dropout",
        "lr",
        "weight_decay",
        "degree_regularization",
        "residual",
        "activation",
    ):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"unmatched configuration: {key}")
    if [c["q"] for c in configs] != [0.0, 0.25]:
        raise ValueError("expected q0 and q025 arms")
    if any(not s["test_evaluated"] or len(s["runs"]) != 10 for s in summaries):
        raise ValueError("both preregistered ten-run experiments must be complete")
    rows = []
    for left, right in zip(summaries[0]["runs"], summaries[1]["runs"], strict=True):
        if (left["seed"], left["split_row"]) != (right["seed"], right["split_row"]):
            raise ValueError("unmatched run seeds/splits")
        directory = f"run-{left['split_row']:02d}-seed-{left['seed']}"
        with (
            np.load(root / "q0" / directory / "split.npz") as a,
            np.load(root / "q025" / directory / "split.npz") as b,
        ):
            if any(
                not np.array_equal(a[key], b[key]) for key in ("train", "val", "test")
            ):
                raise ValueError("unmatched actual saved split arrays")
        rows.append(
            {
                "split": left["split_row"],
                "seed": left["seed"],
                "q0_test_accuracy": left["test"]["accuracy"],
                "q025_test_accuracy": right["test"]["accuracy"],
                "difference_percentage_points": 100
                * (right["test"]["accuracy"] - left["test"]["accuracy"]),
                "q0_best_epoch": left["best_epoch"],
                "q025_best_epoch": right["best_epoch"],
            }
        )
    difference = np.array([row["difference_percentage_points"] for row in rows])
    result = {
        "comparison": "same DHSNN architecture: q=0.25 minus q=0; no test selection",
        "dataset_sha256": configs[0]["dataset_sha256"],
        "source_hashes_match": True,
        "saved_split_arrays_match": True,
        "runs": rows,
        "paired_difference_percentage_points_mean": float(difference.mean()),
        "paired_difference_percentage_points_population_std": float(difference.std()),
        "paired_difference_percentage_points_sample_std": float(difference.std(ddof=1)),
        "q025_wins": int((difference > 0).sum()),
        "ties": int((difference == 0).sum()),
        "q0_wins": int((difference < 0).sum()),
        "summaries_sha256": {
            name: hashlib.sha256(
                (root / name / "summary.json").read_bytes()
            ).hexdigest()
            for name in ("q0", "q025")
        },
    }
    (root / "paired-comparison.json").write_text(json.dumps(result, indent=2), "utf-8")
    with (root / "paired-runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
