"""Check paired reports using independent miniature experiment artifacts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_dhsnn_cora.py"
_SPEC = importlib.util.spec_from_file_location("dhsnn_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORT)


@pytest.fixture
def experiment(tmp_path):
    config = {
        "model": "dhsnn",
        "source_sha256": {"model.py": "same-source"},
        "dataset_sha256": "same-data",
        "runs": 10,
        "seed": 0,
        "epochs": 1500,
        "patience": 200,
        "variant": "diagonal",
        "stalk_dim": 4,
        "hidden_dim": 16,
        "num_layers": 2,
        "input_dropout": 0.0,
        "rank": 1,
        "orth_strategy": "cayley",
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "degree_regularization": 0.0,
        "residual": False,
        "activation": "complex_relu",
    }
    for name, q, accuracy in (("q0", 0.0, 0.8), ("q025", 0.25, 0.7)):
        arm = tmp_path / name
        arm.mkdir()
        (arm / "config.json").write_text(json.dumps({**config, "q": q}), "utf-8")
        runs = []
        for seed in range(10):
            folder = arm / f"run-{seed:02d}-seed-{seed}"
            folder.mkdir()
            np.savez(
                folder / "split.npz",
                train=np.array([0, 1]),
                val=np.array([2]),
                test=np.array([3]),
            )
            runs.append(
                {
                    "seed": seed,
                    "split_row": seed,
                    "test": {"accuracy": accuracy},
                    "best_epoch": seed + 1,
                }
            )
        (arm / "summary.json").write_text(
            json.dumps({"test_evaluated": True, "runs": runs}), "utf-8"
        )
    return tmp_path


def test_matching_report_and_paired_arithmetic(experiment):
    result = _REPORT.summarize(experiment)
    assert result["paired_difference_percentage_points_mean"] == pytest.approx(-10.0)
    assert result["q0_wins"] == 10 and result["q025_wins"] == 0
    assert len(result["runs"]) == 10
    assert (experiment / "paired-runs.csv").is_file()
    saved = json.loads((experiment / "paired-comparison.json").read_text("utf-8"))
    assert saved == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_dropout", 0.2),
        ("rank", 2),
        ("orth_strategy", "householder"),
        ("model", "hsnn"),
    ],
)
def test_reject_mismatched_configuration(experiment, field, value):
    path = experiment / "q025/config.json"
    config = json.loads(path.read_text("utf-8"))
    config[field] = value
    path.write_text(json.dumps(config), "utf-8")
    with pytest.raises(ValueError, match="dhsnn|unmatched configuration"):
        _REPORT.summarize(experiment)
    assert not (experiment / "paired-comparison.json").exists()


def test_reject_matching_non_dhsnn_models(experiment):
    for arm in ("q0", "q025"):
        path = experiment / arm / "config.json"
        config = json.loads(path.read_text("utf-8"))
        config["model"] = "hsnn"
        path.write_text(json.dumps(config), "utf-8")
    with pytest.raises(ValueError, match="models must be dhsnn"):
        _REPORT.summarize(experiment)
